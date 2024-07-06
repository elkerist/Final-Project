import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW

# 加载数据
def load_snli_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if sample['gold_label'] != '-':
                data.append(sample)
    return data

train_file_path = r"C:\Users\LENOVO\Desktop\snli_1.0\snli_1.0\snli_1.0_dev.jsonl"
train_data = load_snli_jsonl(train_file_path)

# 标签到ID的映射
label_to_id = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 数据预处理
def preprocess_data(data, tokenizer, max_length=128):
    inputs = []
    labels = []
    for sample in data:
        encoded1 = tokenizer(sample['sentence1'], return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        encoded2 = tokenizer(sample['sentence2'], return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        inputs.append((encoded1, encoded2))
        labels.append(label_to_id[sample['gold_label']])
    return inputs, labels

train_inputs, train_labels = preprocess_data(train_data, tokenizer)

# 数据集和数据加载器
class SNLIDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input1, input2 = self.inputs[idx]
        label = self.labels[idx]
        return input1, input2, label

train_dataset = SNLIDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# BiAttention 模型
class BiAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiAttention, self).__init__()
        self.linear_A = nn.Linear(input_dim, hidden_dim)
        self.linear_B = nn.Linear(input_dim, hidden_dim)

    def forward(self, A, B):
        A_proj = self.linear_A(A)
        B_proj = self.linear_B(B)
        attention_matrix = torch.bmm(A_proj, B_proj.transpose(1, 2))
        A_weights = F.softmax(attention_matrix, dim=-1)
        B_weights = F.softmax(attention_matrix.transpose(1, 2), dim=-1)
        A_star = torch.bmm(A_weights, B)
        B_star = torch.bmm(B_weights, A)
        return A_star, B_star

# SentenceRelationClassifier 模型
class SentenceRelationClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentenceRelationClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, aggregated_A, aggregated_B):
        combined = torch.cat([aggregated_A, aggregated_B], dim=-1)
        out = self.relu(self.fc1(combined))
        out = self.fc2(out)
        return out
    
def combine_representations(A, A_star, B, B_star):
    # Concatenate the representations
    combined_A = torch.cat([A, A_star, A - A_star, A * A_star], dim=-1)
    combined_B = torch.cat([B, B_star, B - B_star, B * B_star], dim=-1)
    return combined_A, combined_B



def aggregate_representation(combined):
    # Here we use mean pooling
    aggregated = torch.mean(combined, dim=1)
    return aggregated


# 加载 BERT 模型
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 初始化模型
input_dim = 768
hidden_dim = 64
output_dim = 3
bi_attention = BiAttention(input_dim, hidden_dim)
classifier = SentenceRelationClassifier(4 * input_dim, 128, output_dim)

# 优化器
optimizer = AdamW(list(bi_attention.parameters()) + list(classifier.parameters()), lr=2e-5)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs1, inputs2, labels = batch
        input_ids1 = inputs1['input_ids'].squeeze(1)
        input_ids2 = inputs2['input_ids'].squeeze(1)
        attention_mask1 = inputs1['attention_mask'].squeeze(1)
        attention_mask2 = inputs2['attention_mask'].squeeze(1)
        
        # BERT 输出
        outputs1 = bert_model(input_ids1, attention_mask=attention_mask1)
        outputs2 = bert_model(input_ids2, attention_mask=attention_mask2)
        
        cls_embedding1 = outputs1.last_hidden_state
        cls_embedding2 = outputs2.last_hidden_state

        # BiAttention
        A_star, B_star = bi_attention(cls_embedding1, cls_embedding2)

        # Combine and Aggregate
        combined_A, combined_B = combine_representations(cls_embedding1, A_star, cls_embedding2, B_star)
        aggregated_A = aggregate_representation(combined_A)
        aggregated_B = aggregate_representation(combined_B)

        # 分类器前向传播
        logits = classifier(aggregated_A, aggregated_B)
        
        # 计算损失
        loss = criterion(logits, torch.tensor(labels))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
