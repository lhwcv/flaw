import argparse
import os
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from dataset import *
import tensorflow as tf


BATCH_SIZE = 24
NUM_UNITS = None


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 256, 256, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, strides=stride1, activation=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False, kernel_size=3,
                         kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, activation=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, 2)
        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.005, trainable=False)
        opt = tf.train.AdamOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    normal_path = '/home/lhw/Dataset/ali_flaw/clc_data/normal'
    abnormal_path = '/home/lhw/Dataset/ali_flaw/clc_data/abnormal'
    ds = FlawDetect_Stage1_Classify_DataSet(normal_path,abnormal_path)
    pp_mean = [135, 131.4, 138.3]
    if isTrain:
        augmentors = [
            imgaug.Resize((256,256)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 4)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('--load', help='load model for training')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()
    dataset_train = get_data('train')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.05), (82, 0.005), (123, 0.0005), (300, 0.00005)])
        ],
        max_epoch=400,
        steps_per_epoch=1000,
        session_init=SaverRestore(args.load) if args.load else None
    )
    num_gpu = max(get_num_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))