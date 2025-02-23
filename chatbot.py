from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history with truncation
    inputs = tokenizer.encode_plus(
        history_string, input_text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )

    # Generate the response from the model with controlled response length and beam search
    outputs = model.generate(
        **inputs,
        max_length=100,  # Limit the length of generated text
        num_beams=5,     # Use beam search for better quality responses
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        early_stopping=True
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    # Optionally, limit the size of the conversation history to avoid too long history
    if len(conversation_history) > 10:  # Keep the last 5 exchanges (10 elements)
        conversation_history = conversation_history[-10:]
