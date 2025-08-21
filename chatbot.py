import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Auto-download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Sample FAQ data
faqs = [
    {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! What can I do for you?",
    "hey": "Hey! How can I help you?",
    "price": "Our prices vary by product. Could you tell me which product you’re interested in?",
    "hours": "We are open from 9 AM to 8 PM, Monday to Saturday.",
    "location": "We are located at 123 Main Street, New York.",
    "refund": "You can request a refund within 14 days of purchase with a valid receipt.",
    "delivery": "We offer free delivery for orders above $50. Delivery time is usually 3-5 days.",
    "contact": "You can contact us at support@example.com or call +1-234-567-890.",
    "bye": "Thank you for visiting! Have a wonderful day!"
}
]

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join(tokens)

# Preprocess FAQ questions
faq_questions = [preprocess(faq["question"]) for faq in faqs]

# Vectorize FAQs
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_questions)

# Greeting
print("🤖 Hello! I am your virtual assistant.")
print("I can help you with our FAQs. Just type your question.")
print("Type 'bye' or 'exit' to end the conversation.\n")

while True:
    user_input = input("You: ").strip()
    user_input_lower = user_input.lower()

    # Exit conditions
    if user_input_lower in ["bye", "exit"]:
        print("🤖 Thank you for chatting with me. Have a great day!")
        break

    # Casual greetings
    elif user_input_lower in ["hi", "hello", "hey"]:
        print("🤖 Hello! How can I assist you today?")
        continue

    # Match with FAQs
    user_vector = vectorizer.transform([preprocess(user_input)])
    similarity_scores = cosine_similarity(user_vector, faq_vectors)
    best_match_idx = similarity_scores.argmax()

    if similarity_scores[0][best_match_idx] > 0.3:  # Confidence threshold
        print("🤖", faqs[best_match_idx]["answer"])
    else:
        print("🤖 I'm sorry, I don't have an exact answer for that. Please contact support@example.com.")
