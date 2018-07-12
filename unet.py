import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.utils import *
from tensorpack import *


class Unet(ModelDesc):
    def __init__(self, n_classes=1,base_feas=8, training=True):
        self.n_classes = n_classes
        self.base_feas=base_feas
        self.training = training

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 512, 512, 3], name='input'),
                tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')]

    def conv_bn_relu(self, input, num_outputs, name, need_pool=True):
        with tf.variable_scope(name):
            net = input
            for i, n in enumerate(num_outputs):
               net = slim.conv2d(net,n,[3,3],activation_fn=None,scope='conv_{}'.format(i))
               net = slim.batch_norm(inputs=net, decay=0.999, epsilon=1e-3,
                                     is_training=self.training,
                                     activation_fn=tf.nn.relu,scale='bn_relu_{}'.format(i))
            if need_pool:
                pool = slim.max_pool2d(net, [2, 2], scope='pool')
                return net, pool
            return net


    def upconv_concat(self, inputA, inputB, num_outputs, name):
        with tf.variable_scope(name):
            up = slim.conv2d_transpose(inputA, num_outputs, kernel_size=[2, 2], stride=2, scope='up_conv')
            concat = tf.concat([up, inputB], axis=-1, name='concat')
            return concat

    def pred(self,x):
        with slim.arg_scope(extra_conv_arg_scope()):
            x=x/255.0

            conv1, pool1 = self.conv_bn_relu(x, num_outputs=[self.base_feas, self.base_feas], name='b1')
            conv2, pool2 = self.conv_bn_relu(pool1, [2*self.base_feas, 2*self.base_feas], name='b2')
            conv3, pool3 = self.conv_bn_relu(pool2, [4*self.base_feas, 4*self.base_feas], name='b3')
            conv4, pool4 = self.conv_bn_relu(pool3, [8*self.base_feas, 8*self.base_feas], name='b4')
            conv5 = self.conv_bn_relu(pool4, [16*self.base_feas, 16*self.base_feas], name='b5', need_pool=False)


            up6 = self.upconv_concat(conv5, conv4, 8*self.base_feas, name='b6')
            conv6 = self.conv_bn_relu(up6, [8*self.base_feas, 8*self.base_feas], name='b6', need_pool=False)

            up7 = self.upconv_concat(conv6, conv3, 4*self.base_feas, name='b7')
            conv7 = self.conv_bn_relu(up7, [4*self.base_feas, 4*self.base_feas], name='b7', need_pool=False)

            up8 = self.upconv_concat(conv7, conv2, 2*self.base_feas, name='b8')
            conv8 = self.conv_bn_relu(up8, [2*self.base_feas, 2*self.base_feas], name='b8', need_pool=False)

            up9 = self.upconv_concat(conv8, conv1, self.base_feas, name='b9')
            conv9 = self.conv_bn_relu(up9, [self.base_feas, self.base_feas], name='b9', need_pool=False)
            conv10 = slim.conv2d(conv9, self.n_classes, kernel_size=[1, 1],activation_fn=None, scope='conv10')
            pred = tf.add(conv10,0,name='output')
            return pred

    def _build_graph(self, inputs):
      x, gt = inputs
      self.pred=self.pred(x)

      entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(gt,self.pred))

      reg_loss = tf.losses.get_regularization_loss(scope='reg_loss')
      self.cost = tf.add(entropy,reg_loss,name='total_cost')
      tf.summary.scalar('cost',self.cost)

    def test_api(self):
        self._build_graph(self.inputs())
        print(self.pred)

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3*1e-3, trainable=False)
        return tf.train.AdamOptimizer(lr,epsilon=10-3)

if __name__=='__main__':
    net =Unet()
    net.test_api()

