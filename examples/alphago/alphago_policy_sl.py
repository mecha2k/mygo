from dlgo.dataprocess.parallel_processor import GoDataProcessor
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.neuralnet.alphago import alphago_model

from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support
from dotenv import load_dotenv
import h5py
import os
import time


def main():
    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 5000

    start_time = time.time()

    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")

    encoder = AlphaGoEncoder()
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data("train", num_games, use_generator=True)
    test_generator = processor.load_go_data("test", num_games, use_generator=True)

    modelfile = AlphaGo_dir + "/my_alphago_sl_policy.h5"
    if os.path.isfile(modelfile):
        myagent = load_prediction_agent(h5py.File(modelfile, "r"))
        alphago_sl_policy = myagent.model
    else:
        input_shape = (encoder.num_planes, rows, cols)
        alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)
    alphago_sl_policy.summary()

    alphago_sl_policy.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

    epochs = 5
    batch_size = 128
    filepath = AlphaGo_dir + "/alphago_sl_policy_{epoch}.h5"

    callbacks = [
        ModelCheckpoint(
            filepath=filepath,
            monitor="accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            verbose=1,
        )
    ]

    num_gene_samples = generator.get_num_samples()
    num_test_samples = test_generator.get_num_samples()
    print("train samples: ", num_gene_samples)
    print("test samples: ", num_test_samples)

    alphago_sl_policy.fit(
        generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=num_gene_samples / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=num_test_samples / batch_size,
        callbacks=callbacks,
    )

    alphago_sl_policy.evaluate(
        test_generator.generate(batch_size, num_classes), steps=num_test_samples / batch_size,
    )

    alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)
    with h5py.File(modelfile, "w") as sl_agent_out:
        alphago_sl_agent.serialize(sl_agent_out)

    print(f"model file saved: {modelfile}")
    print(f"elapsed time ({encoder.name()}): {time.time() - start_time} sec.")


if __name__ == "__main__":
    freeze_support()
    main()
