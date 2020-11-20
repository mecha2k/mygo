from dlgo.gtp import GTPFrontend
from dlgo.agent.predict import load_prediction_agent
from dlgo.agent import termination

from dotenv import load_dotenv
import os
import h5py

load_dotenv(verbose=True)
AGENT_DIR = os.getenv("AGENT_DIR")
bot_name = AGENT_DIR + "/deep_bot_h5py.h5"

model_file = h5py.File(bot_name, "r")
agent = load_prediction_agent(model_file)
strategy = termination.get("opponent_passes")
termination_agent = termination.TerminationAgent(agent, strategy)

frontend = GTPFrontend(termination_agent)
frontend.run()
