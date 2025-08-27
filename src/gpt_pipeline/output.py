import torch

def line(prompt, dataset, device_model):
    context = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=dataset.device)

    output_tokens = device_model.generate(context, max_new_tokens=300)[0].tolist()
    output_text = dataset.decode(output_tokens)

    # print(prompt)
    print(output_text)