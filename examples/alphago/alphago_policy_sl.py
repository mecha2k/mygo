from dlgo.dataprocess.myprocessor import GoDataProcessor

from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.agent.predict import DeepLearningAgent
from dlgo.neuralnet.alphago import alphago_model

from keras.callbacks import ModelCheckpoint
from multiprocessing import freeze_support
from dotenv import load_dotenv
import h5py
import os


def main():
    rows, cols = 19, 19
    num_classes = rows * cols
    num_games = 100

    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")

    encoder = AlphaGoEncoder()
    processor = GoDataProcessor(encoder=encoder.name())
    generator = processor.load_go_data("train", num_games, use_generator=True)
    test_generator = processor.load_go_data("test", num_games, use_generator=True)

    input_shape = (encoder.num_planes, rows, cols)
    alphago_sl_policy = alphago_model(input_shape, is_policy_net=True)
    alphago_sl_policy.summary()

    alphago_sl_policy.compile("sgd", "categorical_crossentropy", metrics=["accuracy"])

    epochs = 10
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

    alphago_sl_policy.fit_generator(
        generator=generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=test_generator.get_num_samples() / batch_size,
        callbacks=callbacks,
    )

    alphago_sl_agent = DeepLearningAgent(alphago_sl_policy, encoder)

    with h5py.File(AlphaGo_dir + "/alphago_sl_policy.h5", "w") as sl_agent_out:
        alphago_sl_agent.serialize(sl_agent_out)

    alphago_sl_policy.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size,
    )


if __name__ == "__main__":
    freeze_support()
    main()
