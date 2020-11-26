from dlgo.agent.pg import PolicyAgent, load_policy_agent
from dlgo.agent.predict import load_prediction_agent
from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.reinforce.simulate import experience_simulation
from dlgo.reinforce import ValueAgent, load_experience, load_value_agent
from dlgo.neuralnet.alphago import alphago_model
from dlgo.kerasutil import init_gpus

from dotenv import load_dotenv
import h5py
import time
import os


def reinforcelearning(workdir, index):
    alphagofile = workdir + "/my_alphago_sl_policy.h5"

    encoder = SimpleEncoder((19, 19))
    sl_agent = load_prediction_agent(h5py.File(alphagofile, "r"))
    sl_opponent = load_prediction_agent(h5py.File(alphagofile, "r"))
    # sl_agent.model.summary()

    rl_agent_file = workdir + "/my_alphago_rl_policy.h5"
    experience_file = workdir + f"/my_alphago_rl_experience_{index}.h5"
    if os.path.isfile(rl_agent_file):
        print("load policy agent from files...")
        alphago_rl_agent = load_policy_agent(h5py.File(rl_agent_file, "r"))
        experience = load_experience(h5py.File(experience_file, "r"))
    else:
        alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
        opponent = PolicyAgent(sl_opponent.model, encoder)

        num_games = 20
        print("simulate experiences...")
        experience = experience_simulation(num_games, alphago_rl_agent, opponent)
        with h5py.File(experience_file, "w") as exp_out:
            experience.serialize(exp_out)

    alphago_rl_agent.train(experience)

    with h5py.File(rl_agent_file, "w") as rl_agent_out:
        alphago_rl_agent.serialize(rl_agent_out)


def value_network(workdir, num_cases):
    rows, cols = 19, 19
    encoder = SimpleEncoder((rows, cols))
    input_shape = (encoder.num_planes, rows, cols)

    value_file = workdir + "/my_alphago_value.h5"
    if os.path.isfile(value_file):
        print("load value agent from file...")
        alphago_value = load_value_agent(h5py.File(value_file, "r"))
    else:
        alphago_value_network = alphago_model(input_shape)
        alphago_value = ValueAgent(alphago_value_network, encoder)

    for index in range(num_cases):
        experience = load_experience(h5py.File(workdir + f"/my_alphago_rl_experience_{index}.h5", "r"))
        alphago_value.train(experience)

    with h5py.File(value_file, "w") as value_agent_out:
        alphago_value.serialize(value_agent_out)


if __name__ == "__main__":
    load_dotenv(verbose=True)
    AlphaGo_dir = os.getenv("ALPHAGO_DIR")

    init_gpus()
    start_time = time.time()

    num_cases = 3
    for i in range(num_cases):
        print(f"reinforment learning of {i} cases...")
        reinforcelearning(AlphaGo_dir, i)
    print(f"value network learning...")
    value_network(AlphaGo_dir, num_cases)

    print(f"time elapsed {time.time() - start_time} sec...")
