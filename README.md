# Fine-Tuning and Text Generation with `unsloth/Llama-3.2-3B-Instruct`


This project demonstrates how to fine-tune a pre-trained language model named unsloth/Llama-3.2-3B-Instruct using the Hugging Face transformers, trl, and datasets libraries. The notebook walks through all steps, from installing dependencies to fine-tuning the model and performing text generation. Below, are detailed instructions and explanations of each step.

## 1. Create and Access the DigitalOcean Droplet
1.	Log in to your DigitalOcean account.
2.	Create an H100 GPU droplet using the AI/ML-ready image (Ubuntu 22.04).
3.	Note the public IP address of the droplet.
4.	SSH into the droplet:

	`ssh root@<DROPLET_IP>`

## 2. Update System and Install Essential Tools
1.	Update the package manager:

	`apt update && apt upgrade -y`

2.	Install essential tools:

	`apt install -y python3 python3-venv python3-pip git docker.io`

## 3. Set Up a Python Virtual Environment
1.	Create a virtual environment:

	`python3 -m venv venv`

2.	Activate the virtual environment:

	`source venv/bin/activate`

## 4. Clone the GitHub Repository

1.	Clone the repository:

	`git clone https://github.com/jkanalakis/finetuning-llama-model-for-text-generation-using-unsloth.git`

	`cd finetuning-llama-model-for-text-generation-using-unsloth`

## 5. Install Dependencies

1. Upgrade pip:

	`pip install --upgrade pip`

2.	Install dependencies:

	`pip install -r requirements.txt`

3.	Install Jupyter Notebook:

	`pip install notebook`

## 6. Run Jupyter Notebook on the Droplet

1.	Start the Jupyter Notebook server:

	`jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`

2.	Copy the token URL from the terminal output (e.g., http://127.0.0.1:8888/?token=...).

## 7. Set Up HTTP Tunneling on Your MacBook

1.	Open a new terminal on your MacBook.

2.	Create an SSH tunnel (*need to connect over port 8889 when the server already uses 8888*):

	`ssh -L 8889:localhost:8888 root@<DROPLET_IP>`

3.	Access Jupyter Notebook by opening the following URL in your browser:

	`http://127.0.0.1:8889`

## 8. Test the Jupyter Notebook

1.	Open the Jupyter Notebook file (.ipynb) in the Jupyter interface.

2.	Run the cells step-by-step to fine-tune the model and generate outputs.

## 9. Download the Fine-Tuned Model

1.	Compress the fine-tuned model directory:

	`tar -czvf finetuned_model.tar.gz finetuned_model`

2.	Download the compressed file to your MacBook:

	```scp root@<DROPLET_IP>:~/finetuning-llama-model-for-text-generation-using-unsloth/finetuned_model.tar.gz ./```

## 10. Save the Python Environment as a Docker Image

1.	Create a file named **Dockerfile** in the repository directory:

	```
	FROM python:3.10-slim
	WORKDIR /app

	COPY requirements.txt ./
	RUN pip install --no-cache-dir -r requirements.txt

	COPY . ./

	CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

	```

2.	Build the Docker image:

	`docker build -t fine-tuning-env .`

3.	Save the Docker image to a file:

	`docker save fine-tuning-env > fine-tuning-env.tar`

4.	Download the Docker image to your MacBook:

	`scp root@<DROPLET_IP>:~/finetuning-llama-model-for-text-generation-using-unsloth/fine-tuning-env.tar ./`

## 11. Shut Down the Python Virtual Environment

1.	Deactivate the virtual environment:

	`deactivate`

2.	Stop the Jupyter Notebook server (Ctrl+C in the terminal running it).

## 12. Optional: Shut Down the Droplet

1.	To avoid additional costs, power off or destroy the droplet:

	`shutdown now`

### Notes
•	Replace <DROPLET_IP> with the public IP address of your droplet.
•	Ensure you have appropriate permissions to use Docker on your MacBook for reusing the environment.

This guide provides a repeatable workflow for configuring, fine-tuning, and saving your environment efficiently.


### Generate Text
Perform text generation using the fine-tuned model.

```python
input_prompt = "Once upon a time"
inputs = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Results

After fine-tuning, the model generates high-quality text that aligns with the dataset's structure and style. The generated text can be used for various NLP tasks like chatbot development, content creation, and more.

## Customization

You can customize:
- **Model:** Replace `unsloth/Llama-3.2-3B-Instruct` with another pre-trained model.
- **Dataset:** Load a different dataset or preprocess it with your templates.
- **Training Parameters:** Adjust learning rate, batch size, and training steps for different results.

## Contributing

Contributions are welcome! Please submit issues or pull requests for improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
