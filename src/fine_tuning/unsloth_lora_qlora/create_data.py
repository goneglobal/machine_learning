from datasets import Dataset

data = [
    {"instruction": "Tell me a joke about cats",
     "input": "",
     "output": "Why was the cat sitting on the computer? It wanted to keep an eye on the mouse."},

    {"instruction": "Give me a joke about robots",
     "input": "",
     "output": "Why did the robot go to therapy? Because it had too many breakdowns."},

    {"instruction": "Tell me a short riddle",
     "input": "",
     "output": "What has keys but can’t open locks? A piano."},
]

dataset = Dataset.from_list(data)
dataset.save_to_disk("toy_dataset")
print(dataset)