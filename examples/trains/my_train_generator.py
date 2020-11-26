import os
import h5py
from dotenv import load_dotenv

from dlgo.dataprocess.myprocessor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent


def train_generator():
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 100

    encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())

    train_gen = processor.load_go_data("train", num_games, use_generator=True)
    test_gen = processor.load_go_data("test", num_games, use_generator=True)

    load_dotenv(verbose=True)
    AGENT_DIR = os.getenv("AGENT_DIR")
    agent_name = AGENT_DIR + "/my_deep_bot.h5"

    model_file = h5py.File(agent_name, "r")
    agent_model = load_prediction_agent(model_file)
    agent_model.model.summary()

    epochs = 2
    batch_size = 128

    train_num = train_gen.get_num_samples(batch_size, num_classes)
    print(train_num)

    agent_model.model.fit(
        train_gen.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=train_gen.get_num_samples() / batch_size,
        validation_data=test_gen.generate(batch_size, num_classes),
        validation_steps=test_gen.get_num_samples() / batch_size,
    )

    agent_model.model.evaluate(
        test_gen.generate(batch_size, num_classes),
        steps=test_gen.get_num_samples() / batch_size,
    )

    deep_learning_bot = DeepLearningAgent(agent_model.model, encoder)
    deep_learning_bot.serialize(h5py.File(AGENT_DIR + "/my_deep_bot01.h5", "w"))


if __name__ == "__main__":
    train_generator()
