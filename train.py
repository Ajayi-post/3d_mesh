import tensorflow as tf
from utils.utils import *
from utils.models import GCN
from utils.fetcher import *
import os

import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_list', 'Data/train_list.txt', 'Data list.') 
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden', 256, 'Number of units in hidden layer.') 
flags.DEFINE_integer('feat_dim', 963, 'Number of units in feature layer.') 
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')


num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)),
    'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
    'labels': tf.placeholder(tf.float32, shape=(None, 6)),
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], 
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], 
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] 
}
model = GCN(placeholders, logging=True)


data = DataFetcher(FLAGS.data_list)
data.setDaemon(True)
data.start()
config=tf.ConfigProto()

config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())



train_loss = open('record_train_loss.txt', 'a')
train_loss.write('Start training, lr =  %f\n'%(FLAGS.learning_rate))
pkl = pickle.load(open('Data/mesh/info_ellipsoid.dat', 'rb'))
feed_dict = construct_feed_dict(pkl, placeholders)





# weight initialization
def disc_weights_init(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
        elif isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(1, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.view(-1, 1).squeeze(1)



netD = Discriminator().to(device)
disc_weights_init(netD)

volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
print('Using ' + obj + ' Data')
volumes = volumes[..., np.newaxis].astype(np.float)
data = torch.from_numpy(volumes)
data = data.permute(0, 4, 1, 2, 3)
data = data.type(torch.FloatTensor)


# choose loss function
criterion = nn.BCELoss()
criterion2 = nn.MSELoss()

# fake/real labels
real_label = 1
fake_label = 0

# setup optimizers
optD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

errD_all = AverageMeter()



train_number = data.number
for epoch in range(FLAGS.epochs):
	all_loss = np.zeros(train_number,dtype='float32') 
	for iters in range(train_number):
		
		img_inp, y_train, data_id = data.fetch()
		feed_dict.update({placeholders['img_inp']: img_inp})
		feed_dict.update({placeholders['labels']: y_train})

		
		_, dists,out1,out2,out3 = sess.run([model.opt_op,model.loss,model.output1,model.output2,model.output3], feed_dict=feed_dict)
		all_loss[iters] = dists


		optD.zero_grad()

		# train Discriminator with real samples
		label = torch.full((train_number,), real_label, device=device)
		out = netD(real_data)
		errD_real = criterion(out, label)

		# train Discriminator with generated samples
		label_fake = label.clone()
		label_fake.fill_(fake_label)
		out = netD(gen_data.detach())
		errD_fake = criterion(out, label_fake)
		errD = (errD_real + errD_fake) * opt.alpha
		errD.backward()
		optD.step()

		# update Generator
		mean_loss = np.mean(all_loss[np.where(all_loss)])
		if (iters+1) % 128 == 0:
			print 'Epoch %d, Iteration %d'%(epoch + 1,iters + 1)
			print 'Mean loss = %f, iter loss = %f, %d'%(mean_loss,dists,data.queue.qsize())
	
	model.save(sess)
	train_loss.write('Epoch %d, loss %f\n'%(epoch+1, mean_loss))
	train_loss.flush()

data.shutdown()
print 'Training Finished!'
