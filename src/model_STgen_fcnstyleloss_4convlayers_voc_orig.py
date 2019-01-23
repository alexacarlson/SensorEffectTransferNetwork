from util import CITYSCAPES_LABELS
from util import num_classes, cmap_trainId2color, cmap_trainId2id, max_pool, conv2d_relu, conv2d_relu2, upscore, cross_entropy
from glob import glob
from PIL import Image
import tensorflow as tf
import os
import time
import pdb
import scipy
import numpy as np
## for sensor transfer
from augmentfunctions_tf import *

class STnet():
    #def __init__(self, *, sess, config, dataset):
    def __init__(self, *, sess, config, dataset_synth, dataset_real):
        self.sess = sess
        self.dataset_synth = dataset_synth
        self.dataset_real = dataset_real

        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.learning_rate = 2e-5#config.learning_rate
        self.tau = config.tau
        self.phase = config.phase
        self.eval_mean = config.eval_mean

        self.images_height_real = config.height
        self.images_width_real = config.width
        self.images_height_synth = 512#config.height
        self.images_width_synth = 1024#config.width
        self.z_dim = 200
        self.name = config.name
        self.log_weights = config.log_weights
        self.loadflag = config.load_weights_flag

        self.checkpoint_dir = os.path.join(config.checkpoint_dir, self.name)
        self.checkpoint_dir_imgs = os.path.join(config.checkpoint_dir, self.name,'aug_imgs')
        self.checkpoint_num = config.checkpoint_number
        self.global_step = 0
        self.log_dir = os.path.join(config.log_dir, self.name)
        for d in [self.checkpoint_dir, self.checkpoint_dir_imgs, self.log_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        self.results_dir = config.result_dir

        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=25)
        self.sess.run(tf.global_variables_initializer())
        ##
        if self.phase=='augment':
            self.loadflag=True
        elif self.phase=='train':
            self.loadflag=False
        else:
            raise ValueError('Unsupported phase.')
            #
        if self.loadflag:
            ## load in weights, no network training
            checkpoint_name = 'model.ckpt-4'
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, checkpoint_name))
            print("Loaded in %s"%checkpoint_name)
            #

    def build_model(self):
        min_queue_examples = 256
        self.g_zbatch = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'zbatch')
        image_dims_real = [self.images_height_real, self.images_width_real, 3]
        image_dims_synth = [self.images_height_synth, self.images_width_synth, 3]
        #lbl_dims = [self.images_height, self.images_width, 1]
        self.Discr_inputs_real = tf.placeholder(tf.float32, [self.batch_size] + image_dims_real, name='D_images_real')
        self.Discr_inputs_synth = tf.placeholder(tf.float32, [self.batch_size] + image_dims_synth, name='D_images_synth')
        self.Gen_inputs_imgs = tf.placeholder(tf.float32, [self.batch_size] + image_dims_synth, name='G_images')
        #
        with tf.variable_scope('model') as scope:
            #            
            ## sensor transformer augmentation generator
            img_train_aug, window_h, sigmas, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval, delta_S, A, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si, a_transl, b_transl = self.augmentation_generator(self.Gen_inputs_imgs, self.g_zbatch)
            #self.aug_img, window_h, sigmas, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval, delta_S, A, a_transl, b_transl = self.augmentation_generator(img_train_synth, self.Gen_zbatch)
            self.aug_img, self.blurSTparams, self.expSTparams, self.colorSTparams, self.noiseSTparams, self.chromabSTparams = self.augmentation_generator_sampler(self.Gen_inputs_imgs, self.g_zbatch, reuse=True)
            #img_train_aug, self.blurSTparams, self.expSTparams, self.colorSTparams, self.noiseSTparams, self.chromabSTparams = self.augmentation_generator(img_train_synth, self.g_zbatch)  
            #
            ## get style loss
            #scope.reuse_variables()
            conv1_1activ_aug, conv1_2activ_aug, conv2_1activ_aug, conv2_2activ_aug, conv3_1activ_aug, conv3_2activ_aug, conv3_3activ_aug, conv4_1activ_aug, conv4_2activ_aug, conv4_3activ_aug = self.net_synth(img_train_aug, None, get_activ=True)
            scope.reuse_variables()
            conv1_1activ_real, conv1_2activ_real, conv2_1activ_real, conv2_2activ_real, conv3_1activ_real, conv3_2activ_real, conv3_3activ_real, conv4_1activ_real, conv4_2activ_real, conv4_3activ_real = self.net_real(self.Discr_inputs_real, None, get_activ=True)
            ## calculate style loss on the early layers
            self.style_loss = tf.reduce_sum( 
                          tf.square(tf.norm(self.gram_matrix(conv1_1activ_aug) - self.gram_matrix(conv1_1activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv1_2activ_aug) - self.gram_matrix(conv1_2activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv2_1activ_aug) - self.gram_matrix(conv2_1activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv2_2activ_aug) - self.gram_matrix(conv2_2activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv3_1activ_aug) - self.gram_matrix(conv3_1activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv3_2activ_aug) - self.gram_matrix(conv3_2activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv3_3activ_aug) - self.gram_matrix(conv3_3activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv4_1activ_aug) - self.gram_matrix(conv4_1activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv4_2activ_aug) - self.gram_matrix(conv4_2activ_real))) + \
                          tf.square(tf.norm(self.gram_matrix(conv4_3activ_aug) - self.gram_matrix(conv4_3activ_real))) 
                          )
            ## blur constraints to prevent it from going to 0 (which will give nans)
            #self.sigmas_loss = -tf.minimum(tf.reduce_min(sigmas),0)*100000
            ## calc total loss
            self.loss_train = self.style_loss/1e6
            #self.loss_train = self.style_loss/1e6 + self.sigmas_loss
            ##
        with tf.variable_scope('optimizer'):
            self.step = tf.placeholder(tf.float32, [], 'step')
            lr = self.learning_rate * tf.exp(-self.step / self.tau)
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_train)
            #
        tf.summary.scalar('learning rate', lr)

        if self.log_weights:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

    def run_model(self):
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        sum_merged = tf.summary.merge_all()
        ##
        ##
        ## Load in real data
        real_data_filepaths = self.dataset_real.train
        self.num_images_train_real = len(real_data_filepaths)
        num_real_steps = int(np.floor(self.num_images_train_real / self.batch_size))
        randombatch_real = np.arange(num_real_steps*self.batch_size)
        np.random.shuffle(randombatch_real)
        ##
        ## Load in synthetic data
        synth_data_imgfilepaths = self.dataset_synth.train
        self.num_images_train_synth = len(synth_data_imgfilepaths)
        num_synth_steps = int(np.ceil(self.num_images_train_synth / self.batch_size))
        randombatch_synth = np.arange(num_synth_steps*self.batch_size)
        np.random.shuffle(randombatch_synth)
        save_weights_interval = num_synth_steps #self.save_weights_interval
        ##
        print_step = 100#num_steps
        print('Start training with {:d} synthetic images...'.format(self.num_images_train_synth))
        print('Run TensorBoard to monitor the training progress, starting now')
        #
        real_data_batch_counter = 0
        synth_data_batch_counter = 0
        checkpoint = os.path.join(self.checkpoint_dir, 'model.ckpt')
        local_step = 0
        #real_data_batch_counter =0
        #pdb.set_trace()
        if not self.loadflag:
            t_start = time.time()
            for epoch in range(self.num_epochs):
                for k in range(num_synth_steps):
                    start_step_time = time.time()
                    ##
                    ## load in a real data batch to feed in the discriminator
                    if real_data_batch_counter == len(real_data_filepaths):
                        np.random.shuffle(randombatch_real)
                        real_data_batch_counter = 0
                    real_batch_idxs = np.random.choice(num_real_steps, self.batch_size)
                    real_batch_files = [real_data_filepaths[randombatch_real[idx]] for idx in real_batch_idxs]
                    real_batch_images = np.array([scipy.misc.imread(real_batch_file) for real_batch_file in real_batch_files]).astype(np.float32)
                    real_data_batch_counter+=1
                    #
                    ## load in a synthetic data batch to feed in the generator/discriminator
                    if synth_data_batch_counter == len(synth_data_imgfilepaths):
                        np.random.shuffle(randombatch_synth)
                        synth_data_batch_counter = 0
                    synth_batch_idxs = np.random.choice(num_synth_steps, self.batch_size)
                    synth_batch_imgfiles = [synth_data_imgfilepaths[randombatch_synth[idx]] for idx in synth_batch_idxs]
                    #synth_batch_images = np.array([read_img(synth_batch_file, self.images_height, self.images_width, 'bilinear') for synth_batch_file in synth_batch_imgfiles])
                    synth_batch_images = np.array([scipy.misc.imread(synth_batch_file) for synth_batch_file in synth_batch_imgfiles])
                    synth_data_batch_counter+=1
                    ##
                    ## generate uniform noise for the batch of images
                    zbatch = np.random.uniform(-1,1,[self.batch_size, self.z_dim]).astype(np.float32)
                    #
                    ## run the training op
                    self.sess.run(self.train_step, feed_dict={self.step: self.global_step, self.Discr_inputs_real:real_batch_images, self.Gen_inputs_imgs:synth_batch_images, self.g_zbatch: zbatch})
                    #print("step %d took %f:"%(k, time.time()-start_step_time))
                    #
                    ## print and save network training data
                    if self.global_step % print_step == 0:
                        #
                        summary, blurSTparams, expSTparams, colorSTparams, noiseSTparams, chromabSTparams = \
                                                      self.sess.run([sum_merged, self.blurSTparams, self.expSTparams, self.colorSTparams, self.noiseSTparams, self.chromabSTparams], 
                                                      feed_dict={self.step: self.global_step, self.Discr_inputs_real:real_batch_images, self.Gen_inputs_imgs:synth_batch_images, self.g_zbatch: zbatch})
                        writer.add_summary(summary, self.global_step)
                        #print("step %d took %f:"%(k, time.time()-start_step_time))
                        #m, s = divmod(time.time() - t_start, 60)
                        #h, m = divmod(m, 60)
                        loss = self.loss_train.eval({self.step: self.global_step, self.Discr_inputs_real:real_batch_images, self.Gen_inputs_imgs:synth_batch_images, self.g_zbatch:zbatch})
                        #print('Epoch: [{: 6.2f}/{:4d}], Time: [{:02d}:{:02d}:{:02d}], Pixel-wise accuracy: {:.4f}'.format(local_step / num_steps, self.num_epochs, int(h), int(m), int(s), accu))
                        print('Completed: [{: 6.2f}] / [{:4d}] epochs, Loss: {:.6E}'.format(local_step/num_synth_steps, self.num_epochs, loss))
                        ## print the selected parameters for this step
                        chromabP = 'chromab, '+ str(np.squeeze(chromabSTparams))
                        noiseP = 'noise, '+ str(np.squeeze(noiseSTparams))
                        blurP = 'blur, ' + str(np.squeeze(blurSTparams))
                        expP = 'exposure, ' + str(np.squeeze(expSTparams))
                        colorP = 'color, ' + str(np.squeeze(colorSTparams))
                        param_str='\n'.join([chromabP, blurP, expP, noiseP, colorP])
                        print(param_str)
                        ##
                    ## track total training iterations run
                    self.global_step += 1
                    local_step += 1
                ## save network info after each epoch
                ## save the augmentation parameters
                param_file = os.path.join(self.checkpoint_dir, 'fcn8-model-STparams-'+'epoch'+str(epoch)+'-iter'+str(self.global_step))
                chromabP = 'chromab, '+ str(np.squeeze(chromabSTparams))
                noiseP = 'noise, '+ str(np.squeeze(noiseSTparams))
                blurP = 'blur, ' + str(np.squeeze(blurSTparams))
                expP = 'exposure, ' + str(np.squeeze(expSTparams))
                colorP = 'color, ' + str(np.squeeze(colorSTparams))
                param_str='\n'.join([chromabP, blurP, expP, noiseP, colorP])
                ## save the parameters to a txt file
                fobj = open(param_file+'.txt','w')
                fobj.write(param_str)
                fobj.close()
            ## save the network weights after training
            print('Saving `{}`'.format(checkpoint))
            self.saver.save(self.sess, checkpoint, global_step=self.num_epochs)
            ##
        ## once training is done, augment entire dataset
        print("Now Augmenting the full synthetic dataset")
        final_step=0
        synth_data_batch_counter_final = 0
        np.random.shuffle(randombatch_synth)
        #
        for step_ in range(self.num_images_train_synth):
            ## run images through the generator (augment only)
            ## load in a synthetic data batch to feed in the generator/discriminator
            #synth_batch_idxs = np.random.choice(num_synth_steps, self.batch_size)
            #synth_batch_imgfiles = [synth_data_imgfilepaths[randombatch_synth[idx]] for idx in synth_batch_idxs]
            #synth_batch_images = [read_img(synth_batch_file, self.images_height, self.images_width, 'bilinear') for synth_batch_file in synth_batch_imgfiles]
            fname = synth_data_imgfilepaths[step_]
            out_img_ID = os.path.split(synth_data_imgfilepaths[step_])[1].split('.')[0]
            synth_batch_image = np.array([scipy.misc.imread(fname)])#, self.images_height, self.images_width, 'bilinear')])
            if synth_batch_image.shape[1]!=512 or synth_batch_image.shape[2]!=1024:
                continue
                ##
            synth_data_batch_counter_final+=1
            ##
            final_zbatch = np.random.uniform(-1,1,[self.batch_size, self.z_dim]).astype(np.float32)
            aug_imgs, blurSTparams, expSTparams, colorSTparams, noiseSTparams, chromabSTparams = \
                    self.sess.run([self.aug_img, self.blurSTparams, self.expSTparams, self.colorSTparams, self.noiseSTparams, self.chromabSTparams], \
                    feed_dict={self.Gen_inputs_imgs:synth_batch_image, self.g_zbatch: final_zbatch})
            ## save the augmentation information
            param_file = os.path.join(self.checkpoint_dir_imgs, out_img_ID+'-STparams')
            chromabP = 'chromab, ' + str(np.squeeze(chromabSTparams))
            noiseP = 'noise, '+ str(np.squeeze(noiseSTparams))
            blurP = 'blur, ' + str(np.squeeze(blurSTparams))
            expP = 'exposure, ' + str(np.squeeze(expSTparams))
            colorP = 'color, ' + str(np.squeeze(colorSTparams))
            param_str='\n'.join([chromabP, blurP, expP, noiseP, colorP])
            fobj = open(param_file+'.txt','w')
            fobj.write(param_str)
            fobj.close()
            ## save the image
            image_out = np.squeeze(aug_imgs[0])
            out_name = os.path.join(self.checkpoint_dir_imgs, out_img_ID + '_aug.jpg')
            image_out[image_out > 255.0] = 255.0
            image_out[image_out < 0.0] = 0.0
            image_save = Image.fromarray((image_out).astype(np.uint8))
            image_save.save(out_name)

    #def net(self, x, y, x_real, zbatch, training=True):
    def net_synth(self, x, y, training=True, get_activ=False):
        ##
        ## mean subtract the image inputs
        use_eval_mean = (not training and self.eval_mean)
        mean_to_subtract = self.dataset_synth.val_image_channel_mean if use_eval_mean else self.dataset_synth.train_image_channel_mean
        x = tf.subtract(x, mean_to_subtract)
        # VGG16 takes bgr format
        conv1_1 = conv2d_relu2(x[..., ::-1], name='conv1_1', training=False)
        conv1_2 = conv2d_relu2(conv1_1, name='conv1_2', training=False)
        pool1 = max_pool(conv1_2, 'pool1')
        ##
        conv2_1 = conv2d_relu2(pool1, name='conv2_1', training=False)
        conv2_2 = conv2d_relu2(conv2_1, name='conv2_2', training=False)
        ##
        pool2 = max_pool(conv2_2, 'pool2')
        ##
        conv3_1 = conv2d_relu2(pool2, name='conv3_1', training=False)
        conv3_2 = conv2d_relu2(conv3_1, name='conv3_2', training=False)
        conv3_3 = conv2d_relu2(conv3_2, name='conv3_3', training=False)
        pool3 = max_pool(conv3_3, 'pool3')
        ####
        #if get_activ:
        #    return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3
        #    ##
        conv4_1 = conv2d_relu2(pool3, name='conv4_1', training=False)
        conv4_2 = conv2d_relu2(conv4_1, name='conv4_2', training=False)
        conv4_3 = conv2d_relu2(conv4_2, name='conv4_3', training=False)
        #pool4 = max_pool(conv4_3, 'pool4')
        ###
        if get_activ:
            return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3
            ##
        ###
        #conv5_1 = conv2d_relu2(pool4, name='conv5_1', training=False)
        #conv5_2 = conv2d_relu2(conv5_1, name='conv5_2', training=False)
        #conv5_3 = conv2d_relu2(conv5_2, name='conv5_3', training=False)
        ###
        #if get_activ:
        #    return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3
        #    ##        
        #pool5 = max_pool(conv5_3, 'pool5')
        ##
        #fc6 = conv2d_relu(pool5, 'fc6')
        #drop6 = tf.layers.dropout(fc6, training=training, name='drop6')
        ##
        #fc7 = conv2d_relu(drop6, 'fc7')
        #drop7 = tf.layers.dropout(fc7, training=training, name='drop7')
        ##
        #score_fr = tf.layers.conv2d(drop7, num_classes, 1, name='score_fr')
        #upscore2 = upscore(score_fr, 2, 'upscore2')
        ##
        #score_pool4 = tf.layers.conv2d(pool4, num_classes, 1, name='score_pool4')
        #fuse_pool4 = tf.add(upscore2, score_pool4, name='fuse_pool4')
        #upscore_pool4 = upscore(fuse_pool4, 2, 'upscore_pool4')
        ##
        #score_pool3 = tf.layers.conv2d(pool3, num_classes, 1, name='score_pool3')
        #fuse_pool3 = tf.add(upscore_pool4, score_pool3, 'fuse_pool3')
        #logits = upscore(fuse_pool3, 8, 'score')
        ##
        #mask = tf.not_equal(y, 255, name='mask')
        ##
        #with tf.variable_scope('loss'):
        #    logits_masked = tf.boolean_mask(logits, mask[..., 0], name='logits_masked')
        #    y_masked = tf.boolean_mask(y, mask, name='labels_masked')
        #    ## semantic seg loss
        #    fcn_loss = cross_entropy(logits=logits_masked, labels=y_masked)
        #    #
        #    ## calculate total loss
        #    loss = fcn_loss 
        #    ##
        #if training:
        #    return loss
        #else:
        #    with tf.variable_scope('prediction'):
        #        ## generate prediction
        #        pred = tf.expand_dims(tf.argmax(logits, axis=-1), -1, name='full')
        #        pred_masked = tf.boolean_mask(pred, mask, name='masked')
        #        pred_correct = tf.equal(y_masked, tf.cast(pred_masked, tf.int32), name='correct')
        #    ## calculate accuracy of prediction
        #    accu = tf.reduce_mean(tf.cast(pred_correct, tf.float32), name='accuracy')
        #    #
        #    return loss, pred, accu
        #    #return loss, pred, accu, D_real, D_logits_real, D_synth, D_logits_synth

    def net_real(self, x, y, training=True, get_activ=False):
        ##
        ## mean subtract the image inputs
        use_eval_mean = (not training and self.eval_mean)
        mean_to_subtract = self.dataset_synth.val_image_channel_mean if use_eval_mean else self.dataset_real.train_image_channel_mean
        x = tf.subtract(x, mean_to_subtract)
        # VGG16 takes bgr format
        conv1_1 = conv2d_relu2(x[..., ::-1], name='conv1_1', training=False)
        conv1_2 = conv2d_relu2(conv1_1, name='conv1_2', training=False)
        pool1 = max_pool(conv1_2, 'pool1')
        ##
        conv2_1 = conv2d_relu2(pool1, name='conv2_1', training=False)
        conv2_2 = conv2d_relu2(conv2_1, name='conv2_2', training=False)
        ##
        pool2 = max_pool(conv2_2, 'pool2')
        ##
        conv3_1 = conv2d_relu2(pool2, name='conv3_1', training=False)
        conv3_2 = conv2d_relu2(conv3_1, name='conv3_2', training=False)
        conv3_3 = conv2d_relu2(conv3_2, name='conv3_3', training=False)
        pool3 = max_pool(conv3_3, 'pool3')
        ####
        #if get_activ:
        #    return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3
        #    ##
        conv4_1 = conv2d_relu2(pool3, name='conv4_1', training=False)
        conv4_2 = conv2d_relu2(conv4_1, name='conv4_2', training=False)
        conv4_3 = conv2d_relu2(conv4_2, name='conv4_3', training=False)
        #pool4 = max_pool(conv4_3, 'pool4')
        ###
        if get_activ:
            return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3
            ##
        ###
        #conv5_1 = conv2d_relu2(pool4, name='conv5_1', training=False)
        #conv5_2 = conv2d_relu2(conv5_1, name='conv5_2', training=False)
        #conv5_3 = conv2d_relu2(conv5_2, name='conv5_3', training=False)
        ###
        #if get_activ:
        #    return conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3
        #    ##        
        #pool5 = max_pool(conv5_3, 'pool5')
        ##
        #fc6 = conv2d_relu(pool5, 'fc6')
        #drop6 = tf.layers.dropout(fc6, training=training, name='drop6')
        ##
        #fc7 = conv2d_relu(drop6, 'fc7')
        #drop7 = tf.layers.dropout(fc7, training=training, name='drop7')
        ##
        #score_fr = tf.layers.conv2d(drop7, num_classes, 1, name='score_fr')
        #upscore2 = upscore(score_fr, 2, 'upscore2')
        ##
        #score_pool4 = tf.layers.conv2d(pool4, num_classes, 1, name='score_pool4')
        #fuse_pool4 = tf.add(upscore2, score_pool4, name='fuse_pool4')
        #upscore_pool4 = upscore(fuse_pool4, 2, 'upscore_pool4')
        ##
        #score_pool3 = tf.layers.conv2d(pool3, num_classes, 1, name='score_pool3')
        #fuse_pool3 = tf.add(upscore_pool4, score_pool3, 'fuse_pool3')
        #logits = upscore(fuse_pool3, 8, 'score')
        ##
        #mask = tf.not_equal(y, 255, name='mask')
        ##
        #with tf.variable_scope('loss'):
        #    logits_masked = tf.boolean_mask(logits, mask[..., 0], name='logits_masked')
        #    y_masked = tf.boolean_mask(y, mask, name='labels_masked')
        #    ## semantic seg loss
        #    fcn_loss = cross_entropy(logits=logits_masked, labels=y_masked)
        #    #
        #    ## calculate total loss
        #    loss = fcn_loss 
        #    ##
        #if training:
        #    return loss
        #else:
        #    with tf.variable_scope('prediction'):
        #        ## generate prediction
        #        pred = tf.expand_dims(tf.argmax(logits, axis=-1), -1, name='full')
        #        pred_masked = tf.boolean_mask(pred, mask, name='masked')
        #        pred_correct = tf.equal(y_masked, tf.cast(pred_masked, tf.int32), name='correct')
        #    ## calculate accuracy of prediction
        #    accu = tf.reduce_mean(tf.cast(pred_correct, tf.float32), name='accuracy')
        #    #
        #    return loss, pred, accu
        #    #return loss, pred, accu, D_real, D_logits_real, D_synth, D_logits_synth

    def augmentation_generator_sampler(self, x, zbatch, reuse=True):
        #
        with tf.variable_scope("gen_aug") as scope:
            if reuse:
                scope.reuse_variables()
            ## blur augmentation
            x, window_h, sigmas = self.blur_generator(x, zbatch, self.batch_size)
            #
            ## chromatic aberration augmentation
            x, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval = self.chromab_generator(x, zbatch, self.batch_size)
            #
            ## exposure augmentation
            x, delta_S, A = self.exp_generator(x, zbatch, self.batch_size)
            #
            ## sensor noise augmentation
            x, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si = self.noise_generator(x, zbatch, self.batch_size)
            #x, mask = self.noise_mask_generator(x, zbatch, self.batch_size)
            #
            ## color augmentation
            x, a_transl, b_transl = self.color_generator(x, zbatch, self.batch_size)
            #
            ## record parameters
            blurSTparams = tf.concat([window_h, sigmas],axis=1)
            #blurSTparams = tf.constant('None',tf.string)
            #
            expSTparams = tf.concat([delta_S, A],axis=1)
            #expSTparams = tf.constant('None',tf.string)
            #
            colorSTparams = tf.concat([a_transl, b_transl],axis=1) 
            #colorSTparams = tf.constant('None',tf.string)
            ##
            noiseSTparams = tf.concat([Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si],axis=1)
            #noiseSTparams = tf.constant('None',tf.string)
            #
            chromabSTparams = tf.concat([scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval],axis=1)
            #chromabSTparams = tf.constant('None',tf.string)
        #
        return x, blurSTparams, expSTparams, colorSTparams, noiseSTparams, chromabSTparams
        #return x, blurSTparams, expSTparams, colorSTparams, noiseSTparams, chromabSTparams, mask

    def augmentation_generator(self, x, zbatch, reuse=False):
        #
        with tf.variable_scope("gen_aug") as scope:
            if reuse:
                scope.reuse_variables()
            ## blur augmentation
            x, window_h, sigmas = self.blur_generator(x, zbatch, self.batch_size)
            #
            ## chromatic aberration augmentation
            x, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval = self.chromab_generator(x, zbatch, self.batch_size)
            #
            ## exposure augmentation
            x, delta_S, A = self.exp_generator(x, zbatch, self.batch_size)
            #
            ## sensor noise augmentation
            x, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si = self.noise_generator(x, zbatch, self.batch_size)
            #x, mask = self.noise_mask_generator(x, zbatch, self.batch_size)
            #
            ## color augmentation
            x, a_transl, b_transl = self.color_generator(x, zbatch, self.batch_size)
            #
        #return x, window_h, sigmas, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval, delta_S, A, a_transl, b_transl
        return x, window_h, sigmas, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval, delta_S, A, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si, a_transl, b_transl


    def blur_generator(self, rgb, zbatch, batch_size, reuse=False):
        ## Sensor transformer blur Augmentation Generator definition
        with tf.variable_scope('blur_aug_gen') as scope:
            if reuse:
                scope.reuse_variables()
            ## parameter generator
            #h1 = tf.contrib.layers.fully_connected(zbatch,200,scope='g_blur_h1_sig',weights_initializer=tf.random_normal_initializer(stddev=0.002),activation_fn=None)  
            #h2 = tf.contrib.layers.fully_connected(zbatch,70,scope='g_blur_h2_sig',
            #                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
            #                                                activation_fn=tf.nn.sigmoid)  
            h3 = tf.contrib.layers.fully_connected(zbatch, 25, scope='g_blur_h3_sig',
                                                              weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                              activation_fn=tf.nn.relu)  
            hout_sigmas = tf.contrib.layers.fully_connected(h3, 1, scope='g_blur_h4_sig',
                                                                  weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                                  activation_fn=tf.nn.relu)  
            bsigma = tf.get_variable('g_blur_sigma_bias',[batch_size,1],initializer=tf.constant_initializer(1.0),trainable=True)
            #bwindow = tf.get_variable('g_blur_window_bias',[1,1],initializer=tf.constant_initializer(1.0),trainable=True)
            #hout_windows, hout_sigmas = tf.split(hout_, 2, axis = 1)
            #
            ## add realistic constraints to the predicted parameters
            #windows_= hout_windows + bwindow
            windows_= tf.constant(7.0,shape=[self.batch_size,1]) #+ bwindow + tf.constant(0.0001)
            #windows_ =  2.0*windows +  tf.ones_like(windows)
            sigmas_ = hout_sigmas + bsigma #+ tf.constant(0.0001)
            #sigmas_ = hout_sigmas + tf.constant(1.0)#tf.constant(0.001)
            ## add the right dimensions to the generated parameters to be batchsizex1x1x1, i.e., [batch_size, width, height, channels]
            windows_h = tf.expand_dims(tf.expand_dims(windows_,2),3)
            sigmas = tf.expand_dims(tf.expand_dims(sigmas_,2),3)
            #window_h = tf.random_uniform((batch_size,1,1,1), minval=3.0, maxval=11.0,dtype=tf.float32)
            #sigmas = tf.random_uniform((batch_size,1,1,1), minval=0.0, maxval=3.0,dtype=tf.float32) # uniform from 0 to 1.5
            #
            ## perform blur augmentation on image batch
            aug_rgb = aug_blur2(rgb, windows_h, sigmas, batch_size)
            #
            return aug_rgb, windows_h, sigmas

    def exp_generator(self, rgb, zbatch, batch_size, reuse=False):
        ## Sensor transformer exposure Augmentation Generator definition
        with tf.variable_scope('exp_aug_gen') as scope:
            if reuse:
                scope.reuse_variables()
            ## parameter generator 
            #h1 = tf.contrib.layers.fully_connected(zbatch,70,scope='g_exp_h_lin1',
            #                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
            #                                                activation_fn=tf.nn.sigmoid)
            h2 = tf.contrib.layers.fully_connected(zbatch, 25, scope='g_exp_h_lin2',
                                                               weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                               activation_fn=tf.nn.relu)
            hout = tf.contrib.layers.fully_connected(h2, 1, scope='g_exp_h_lin3',
                                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                                activation_fn=None)  
            #bias_exp_A = tf.get_variable('g_exp_bias_A',[1],initializer=tf.constant_initializer(0.001))
            bias_exp_ds = tf.get_variable('g_exp_bias_ds',[1],initializer=tf.constant_initializer(0.001))
            #hout_ = hout + bias_exp
            #delta_S, A = tf.split(hout, 2, axis = 1)
            delta_S = hout + bias_exp_ds #+ tf.constant(0.0001)
            #delta_S = hout + tf.constant(0.001)
            #A = tf.constant(0.85, shape=[batch_size,1]) + bias_exp_A + tf.constant(0.0001)
            A = tf.constant(0.85, shape=[batch_size,1]) 
            #delta_S = hout
            ## add the right dimensions to the generated parameters to be batchsizex1x1x1, i.e., [batch_size, width, height, channels]
            delta_S_ = tf.expand_dims(tf.expand_dims(delta_S,2),3)
            A_ = tf.expand_dims(tf.expand_dims(A,2),3)
            #
            ## perform exposure augmentation on image batch
            aug_rgb = aug_exposure(rgb, delta_S_, A_, batch_size)
            #
            return aug_rgb, delta_S, A

    def color_generator(self, rgb, zbatch, batch_size, reuse=False):
        ## Sensor transformer Color Augmentation Generator definition
        with tf.variable_scope('color_aug_gen') as scope:
            if reuse:
                scope.reuse_variables()
            ## color parameter generator
            #h1 = tf.contrib.layers.fully_connected(zbatch,75,scope='g_col_h_lin1',
            #                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
            #                                                activation_fn=tf.nn.sigmoid)
            h2 = tf.contrib.layers.fully_connected(zbatch, 25, scope='g_col_h_lin2',
                                                               weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                               activation_fn=tf.nn.relu)
            hout = tf.contrib.layers.fully_connected(h2, 2, scope='g_col_h_lin3',
                                                            weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                            activation_fn=None)  
            bias_color = tf.get_variable('g_color_bias',[2],initializer=tf.constant_initializer(0.001))
            hout_ = hout + bias_color
            #bias_color_a = tf.get_variable('g_color_bias_a',[1],initializer=tf.constant_initializer(1.0))
            #bias_color_b = tf.get_variable('g_color_bias_b',[1],initializer=tf.constant_initializer(1.0))
            #hout_ = hout + tf.constant(0.0001)
            a_transl, b_transl = tf.split(hout_, 2, axis = 1)
            a_transl_ = a_transl #+ tf.constant(0.0001) + bias_color_a 
            b_transl_ = b_transl #+ tf.constant(0.0001) + bias_color_b
            ## add the right dimensions to the generated parameters to be batchsizex1x1x1, i.e., [batch_size, width, height, channels]
            a_transl__ = tf.expand_dims(tf.expand_dims(a_transl_, 2), 3)
            b_transl__ = tf.expand_dims(tf.expand_dims(b_transl_, 2), 3)
            #
            ## perform color augmentation on image batch
            aug_rgb = aug_color(rgb, a_transl__, b_transl__)
            #
            return aug_rgb, a_transl, b_transl


    def noise_generator(self, rgb, zbatch, batch_size, reuse=False):
        ## Sensor transformer Sensor Noise Augmentation Generator definition
        with tf.variable_scope('noise_aug_gen') as scope:
            if reuse:
                scope.reuse_variables()
                #
            ### parameter generator for all channels
            #h1 = tf.contrib.layers.fully_connected(zbatch,75,scope='gRGB_noise_hlin1',
            #                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
            #                                                activation_fn=tf.nn.relu)
            h2 = tf.contrib.layers.fully_connected(zbatch, 25, scope='gRGB_noise_hlin2',
                                                               weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                               activation_fn=tf.nn.relu)  
            hout = tf.contrib.layers.fully_connected(h2, 6, scope='gRGB_noise_hlin3',
                                                            weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                            activation_fn=tf.nn.relu)  
            #bias_noise = tf.get_variable('g_noise_bias',[1],initializer=tf.constant_initializer(0.0001))
            #bias_noise_R = tf.get_variable('g_noise_bias_R',[1],initializer=tf.constant_initializer(0.0001))
            #bias_noise_G = tf.get_variable('g_noise_bias_G',[1],initializer=tf.constant_initializer(0.0001))
            #bias_noise_B = tf.get_variable('g_noise_bias_B',[1],initializer=tf.constant_initializer(0.0001))
            #hout_ = hout + bias_noise #+ tf.constant(0.0001)
            hout_ = hout + tf.constant(0.00001) #+ tf.random_normal([1], mean=0.0001, stddev=0.00005)
            Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si = tf.split(hout_, 6, axis = 1)
            #a_sd, b_si = tf.split(hout, 2, axis = 1)
            #
            ## add the right dimensions to the generated parameters to be batchsizex1x1x1, i.e., [batch_size, width, height, channels]
            Ra_sd =  tf.expand_dims(tf.expand_dims( Ra_sd,2),3)
            Rb_si =  tf.expand_dims(tf.expand_dims( Rb_si,2),3)
            Ga_sd =  tf.expand_dims(tf.expand_dims( Ga_sd,2),3)
            Gb_si =  tf.expand_dims(tf.expand_dims( Gb_si,2),3) 
            Ba_sd =  tf.expand_dims(tf.expand_dims( Ba_sd,2),3)
            Bb_si =  tf.expand_dims(tf.expand_dims( Bb_si,2),3)
            #
            ## perform augmentation on image
            aug_rgb = aug_noise(rgb, batch_size, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si, self.images_height_synth, self.images_width_synth)
            #
            return aug_rgb, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si

    def chromab_generator(self, rgb, zbatch, batch_size, reuse=False):
        ## Sensor transformer Chromatic Aberration Augmentation Generator definition
        with tf.variable_scope('chromab_aug_gen') as scope:
            if reuse:
                scope.reuse_variables()
            ## parameter generator
            #h1 = tf.contrib.layers.fully_connected(zbatch,75,scope='g_chromab_hlin1',weights_initializer=tf.random_normal_initializer(stddev=0.002),activation_fn=tf.nn.sigmoid)
            h2 = tf.contrib.layers.fully_connected(zbatch,25,scope='g_chromab_hlin2',weights_initializer=tf.random_normal_initializer(stddev=0.002),activation_fn=tf.nn.relu)
            hout = tf.contrib.layers.fully_connected(h2, 7, scope='g_chromab_h_lin3',
                                                                weights_initializer=tf.random_normal_initializer(stddev=0.002),
                                                                activation_fn=None)  
            #bias_chromab = tf.get_variable('g_chromab_bias',[1],initializer=tf.constant_initializer(0.00001))
            #hout = hout + bias_chromab
            scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval = tf.split(hout, 7, axis = 1)
            scale_val_ = scale_val + tf.ones_like(scale_val) #+ tf.constant(0.00001) #+ bias_chromab 
            tx_Rval_ = tx_Rval #+ tf.constant(0.00001) #+ bias_chromab
            tx_Gval_ = tx_Gval #+ tf.constant(0.00001) #+ bias_chromab
            tx_Bval_ = tx_Bval #+ tf.constant(0.00001) #+ bias_chromab
            ty_Rval_ = ty_Rval #+ tf.constant(0.00001) #+ bias_chromab
            ty_Gval_ = ty_Gval #+ tf.constant(0.00001) #+ bias_chromab
            ty_Bval_ = ty_Bval #+ tf.constant(0.00001) #+ bias_chromab
            ## add the right dimensions to the generated parameters to be batchsizex1x1x1, i.e., [batch_size, width, height, channels]
            scale_val = tf.expand_dims(tf.expand_dims( scale_val_,2),3)# tf.expand_dims(tf.expand_dims(tf.constant(1.0,shape = (batch_size,1)),2),3)
            tx_Rval =  tf.expand_dims(tf.expand_dims( tx_Rval_,2),3)
            ty_Rval =  tf.expand_dims(tf.expand_dims( ty_Rval_,2),3) 
            tx_Gval =  tf.expand_dims(tf.expand_dims( tx_Gval_,2),3) 
            ty_Gval =  tf.expand_dims(tf.expand_dims( ty_Gval_,2),3) 
            tx_Bval =  tf.expand_dims(tf.expand_dims( tx_Bval_,2),3) 
            ty_Bval =  tf.expand_dims(tf.expand_dims( ty_Bval_,2),3)
            aug_rgb = aug_chromab(rgb, self.images_height_synth, self.images_width_synth, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval)
            #
            return aug_rgb, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval

    def gram_matrix(self, feature_maps):
        """Computes the Gram matrix for a set of feature maps."""
        batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
        denominator = tf.to_float(height * width)
        feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
        matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
        return matrix / denominator 

    def load(self):
        checkpoint_info_str = 'checkpoint {}'.format(self.checkpoint_num) if self.checkpoint_num else 'latest checkpoint'
        print(' [*] Reading {} from `{}` ...'.format(checkpoint_info_str, self.checkpoint_dir))

        checkpoint_state = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if self.checkpoint_num:
            print("self.checkpoint_num: {}".format(self.checkpoint_num))
            valid_checkpoints = [cp.split('-')[-1] for cp in checkpoint_state.all_model_checkpoint_paths]
            if self.checkpoint_num not in valid_checkpoints:
                raise ValueError("checkpoint num {} not found, valid checkpoints are: {}\nfor files {}".format(
                    self.checkpoint_num,
                    valid_checkpoints,
                    list(checkpoint_state.all_model_checkpoint_paths)
                ))
            checkpoint_path = '-'.join(
                checkpoint_state.model_checkpoint_path.split('-')[:-1] +
                [self.checkpoint_num])
        elif checkpoint_state:
            checkpoint_path = checkpoint_state.model_checkpoint_path
        else:
            checkpoint_path = None

        if checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_path)

            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, checkpoint_name))
            self.global_step = int(checkpoint_name.split('-')[-1])
            print(' [*] Success reading {}. Starting at step {}'.format(checkpoint_name, self.global_step))
        else:
            if self.phase == 'train':
                print(' [*] Failed to find a checkpoint, starting at step 0.')
                assert self.global_step == 0
            else:
                raise ValueError(' [*] Failed to find a checkpoint')
