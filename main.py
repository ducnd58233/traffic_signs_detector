from my_utils import split_data, order_test_set, create_generators
from deeplearning_models import NNModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

if __name__ == "__main__":
    PREPROCESS = False
    TRAIN = True
    TEST = True

    if PREPROCESS:
        path_to_data = "./dataset/Train"
        path_to_save_train = "./dataset/training_data/train"
        path_to_save_val = "./dataset/training_data/val"
        split_data(path_to_data, path_to_save_train, path_to_save_val)
    
        path_to_images = "./dataset/Test"
        path_to_csv = "./dataset/Test.csv"
        order_test_set(path_to_images, path_to_csv)
   
    path_to_train = "./dataset/training_data/train"
    path_to_val = "./dataset/training_data/val"
    path_to_test = "./dataset/Test"
    batch_size = 64
    epochs = 15
    lr = 0.001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    if TRAIN:
        # Save the best model
        path_to_save_model = "./Models"
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_freq="epoch",
            verbose=1 # debug to check if model is saved or not
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=3) # after n patience if val acc not go up it will stop
        optimer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            amsgrad=True
        )

        model = NNModel(nbr_classes)
        model.compile(optimizer=optimer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            train_generator,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop] 
        )


    if TEST:
        model = tf.keras.models.load_model("./Models")
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set:")
        model.evaluate(test_generator)




