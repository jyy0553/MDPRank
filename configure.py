from tensorflow import flags


flags.DEFINE_float("dropout_keep_prob",0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("learning_rate", 0.00001, "learn rate( default: 0.0)")

flags.DEFINE_string('data','OHSUMED','data set')
# flags.DEFINE_string("file_name","Fold1","current_file_name")
flags.DEFINE_string("file_name","Fold5","current_file_name")
# Training parameters
# flags.DEFINE_integer("batch_size", 320, "Batch Size (OHSUMED dataset)")
flags.DEFINE_integer("feature_dim", 25, "feature size")

flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs (default: 200)")

flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")

flags.DEFINE_float('reward_decay',0.99,'reward_decay')


# flags.DEFINE_string('CNN_type','ircnn','data set')

flags.DEFINE_float('sample_train',1,'sampe my train data')
flags.DEFINE_boolean('fresh',True,'wheather recalculate the embedding or overlap default is True')
# flags.DEFINE_string('pooling','max','pooling strategy')
flags.DEFINE_boolean('clean',False,'whether clean the data')
# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#data_helper para

# flags.DEFINE_boolean('isEnglish',True,'whether data is english')
