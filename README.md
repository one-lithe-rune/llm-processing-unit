#  ![a logo reminiscent of a diagram of a simple circuit or complex piping coloured in red, yellow, blue and white on a black background](./docs/images/256x256-logo.png) &nbsp; llm-processing-unit

A small Python library by [one-lithe-rune](https://github.com/one-lithe-rune/) that treats a connection to an instruct trained Large Language Model (LLM) as if it were somewhat analogous to the Arthimetic and Logic Unit of a very simple 8-Bit style processor (think [6502](https://en.wikipedia.org/wiki/MOS_Technology_6502)) of yore. Instead of supporting arithmetic operations such as addition and subraction, it supports common and basic LLM operations such as sending instructions, maintaining contexts and storing responses for further processing.

## Motivation

I needed to organise and refactor my mess of code for talking to LLMs into something that was actually coherent with a driving principle (however mad) behind it, whilst studiously ignoring the obvious "Why not just use [langchain](https://github.com/langchain-ai/langchain) like a normal person?" question.

I needed something small, clear and basic that would let me clarify my own ideas about *what* to use all this fancy AI stuff for, but also give me something to handle the *how* to use it more cleanly. So a learning and mental throat clearing exercise basically, but one I'm able to use as the basis for other stuff.

## What's New

- 16-May-2024: Everything.

## Installation

It's just this git repo for now, no [pypi](https://pypi.org/) package yet. So clone the repo or add it as a submodule to your project and `pip install -r` the `requirements.txt` into your python [virtual environment](https://docs.python.org/3/library/venv.html). You did make one, right?

## Usage Example

```Python

# You're obviously going to need a connection to an LLM of some kind.
# This example is for a local server running on port 5001 that can
# speak the OpenAI Chat protocol using REST. This is the only server
# protocol I've implemented so far.

# The LLM you're using will want things in some particular format,
# I've implemented Alpaca, Llama 3 Chat, Llama 3 Character Chat,
# Llama 3 base and Open AI Chat. Only the first three are much tested.

from llmpu.formatters import Llama3InstructSessionFormatter
from llmpu.sessions import OAICompatibleChatSession

endpoint = OAICompatibleChatSession(
    host="http://localhost:5001/",
    initial_processors=[Llama3InstructSessionFormatter],
    extra_props={ "temperature": 0.7 },
)

# The processing unit is implemented by the LlmProcessingUnit class
from llmpu import LlmProcessingUnit

llm = LlmProcessingUnit(
    endpoint,
    context_registers=3     # will be context0, context1, context2
)

# load a system prompt into the System register
llm.load_sys(
    "You are an unhelpful and sarcastic AI fanfic writing assistant obsessed with llamas."
)

# load an instruction
llm.load_ins(
    "Write a story that doesn't include any llamas."
)

# Store the instruction at a memory location. Memory is just a
# dictionary, with memory locations specified by paths through the
# dictionary. Each memory location holds a LIFO stack.
llm.push("instruction", ["instructions"])

# sends the contents of the system, context0 and, instruction
# registers to the LLM. The response goes into the result register,
# pass in a list of registers if you want a different set or order
llm.evaluate()

# get the contents from the result register
if "llama" in llm.read_result().content
    location = "llamas"
else
    location = "no llamas"

# We want the correct memory location to containing both
# the instruction and the our last result
llm.push("instruction", [location, "transcript"])
llm.push("result", [location, "transcript"])

# We can save or load the entire memory to a json file
# with save_mem and load_mem at any point
llm.save_mem("./maybe_llamas.json")

if location == "llamas":
    print("llamas detected!")
    sys.exit(1)

# load the first context register, which includes what we just
# sent as an instruction and the reponse, ready for the next
# evaluation

llm.load_context(0, ["llamas", "transcript"])
llm.load_ins("change the setting to be IN SPACE!")
llm.evaluate()

# ...see comments in ./llmpu/llmpu.py for more information
```

## Obligatory Chatbot Example

If you're cloned the repo, activated your venv and installed the requirements, you should be able to run the obligatory ChatBot example by doing `python -m llmpu.examples.chat`. See the connection options by doing `python -m llmpu.examples.chat --help`

## What's 'Supported'

- Python 3.11
- As much of the OpenAI chat endpoint protocol sufficient to work against a compatible local AI server endpoint in non-streaming mode but not necessarily enough for a real OpenAI endpoint.
- Formatters for [Alpaca](https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#data-release), [Llama 3 Chat](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3), Llama 3 Character Chat, [Llama 3 Base](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3) and Open AI Chat formats. Only the first three can be considered tested.

I've been developing this against my local instance of [koboldcpp](https://github.com/LostRuins/koboldcpp)/[koboldcpp-rocm](https://github.com/YellowRoseCx/koboldcpp-rocm/) on Linux with various GGUF quantised [Llama 3 8B](https://lama.meta.com/docs/get-started/) variants. So that *should* work.
