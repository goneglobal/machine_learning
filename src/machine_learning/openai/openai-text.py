from openai import OpenAI

# Example function to generate text using OpenAI's API
def generate_text(prompt, model="text-davinci-003", max_tokens=150):
    client = OpenAI()
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()
  
if __name__ == "__main__":
    prompt = "Explain the theory of relativity in simple terms."
    generated_text = generate_text(prompt)
    print("Generated Text:\n", generated_text)