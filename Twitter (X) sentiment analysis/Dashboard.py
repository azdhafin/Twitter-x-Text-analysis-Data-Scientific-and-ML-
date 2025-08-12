import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 6  # Adjust based on your dataset
model = BERTClassifier("bert-base-uncased", num_classes)
model.load_state_dict(torch.load("saved_model.pt", map_location=device))
model.to(device)
model.eval()

st.title("Cyberbullying Detection Dashboard")
st.write("Enter a tweet below to classify its cyberbullying type.")

# Input field
user_input = st.text_area("Tweet Text", height=100)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        labels = {
            0: "Age",
            1: "Ethnicity",
            2: "Gender",
            3: "Not Cyberbullying",
            4: "Other Type",
            5: "Religion"
        }

        st.subheader("Prediction Result:")
        st.success(f"Predicted Category: **{labels[predicted_class]}**")

        st.subheader("Confidence Scores:")
        for i, prob in enumerate(probs[0]):
            st.write(f"{labels[i]}: {prob.item():.4f}")
