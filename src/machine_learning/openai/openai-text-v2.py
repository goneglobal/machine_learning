from openai import OpenAI

# Example function to generate text using OpenAI's API
def generate_text(prompt):
    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o",
        prompt=prompt
    )
    return response.output_text
  
if __name__ == "__main__":
    prompt = "Explain the theory of relativity in simple terms."
    generated_text = generate_text(prompt)
    print("Generated Text:\n", generated_text)