from sklearn import metrics
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from tqdm import tqdm

class AraBERTDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, model_name):
        input_ids = []
        attention_masks = []
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for i in range(len(x)):
            # tokenize each sentence using bert's tokenizer
            # the tokenizer returns a batch encoding, which is derived from a dictionary
            # this dictionary holds the various inputs needed by the model of such tokenizer.
            encoded_sentence = tokenizer.encode_plus(
                    text=x[i],                      # Preprocess sentence
                    add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                    max_length=64,                  # Max length to truncate/pad
                                                    # The model is trained on a sequence length of 64, using max length beyond 64 might result in degraded performance
                    padding='max_length',           # Pad sentence to max length
                    return_attention_mask=True,     # Return attention mask
                    truncation = True)
            input_ids.append(encoded_sentence.get('input_ids'))
            attention_masks.append(encoded_sentence.get('attention_mask'))
        
        # change lists to torch tensors
        self.input_ids = torch.tensor(input_ids)
        self.attention_masks = torch.tensor(attention_masks)
        self.labels = torch.tensor(y)
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

#############################################
class BertClassifier(torch.nn.Module):
  def __init__(self, model_name, hidden_size=50, n_classes=3, freeze_bert=True):
    
    super(BertClassifier, self).__init__()
    # Create the layers of the model
    
    # Retrieve instance from bert model
    self.bert = AutoModel.from_pretrained(model_name)

    # Create the classifier head.
    self.classifier_head = torch.nn.Sequential(
      # Size of input to the first linear layer is 768 nodes
      # Because by printing the bert model, you'll find that the out_features (# of nodes) of the last layer in bert is 768
      torch.nn.Linear(in_features=768, out_features=hidden_size),
      torch.nn.ReLU(),
      torch.nn.Dropout(0.5),
      torch.nn.Linear(in_features=hidden_size, out_features=n_classes)
    )

    # Freeze bert and only train classifier head
    if freeze_bert:
      for params in self.bert.parameters():
          params.requires_grad = False
    
    #####################################################################################################

  def forward(self, input_ids, attention_mask):
    """
    This function does the forward pass of our model
    """
    output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #  pooler_output: Last layer hidden-state of the first token of the sequence (classification token) 
    # pooler_output: (batch_size, 768)
    output = self.classifier_head(output.pooler_output)
    ###############################################################################################
    return output

###########################


def get_bert_embeddings(X,y, model_name='aubmindlab/bert-base-arabertv02-twitter'):
  arabert_prep = ArabertPreprocessor(model_name=model_name)
  X = X.apply(arabert_prep.preprocess)
  # instantiate train and validation datasets
  dataset = AraBERTDataset(X, y, model_name)
  dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
  extractor_model = AutoModel.from_pretrained(model_name)
  embeddings=[]
  for input_id,mask,_ in tqdm(dataloader):
    # extract embeddings of the batch
    output = extractor_model(input_ids=input_id, attention_mask=mask)
    embedding = output.pooler_output
    # append batch embeddings to the total data embeddings
    embeddings += embedding.tolist()
  return embeddings

############################

def train(model, train_dataset, val_dataset, criterion, optimizer, classes_names, n_classes=3, batch_size=16, epochs=30):
  """
  This function implements the training logic
  Inputs:
  - model: the model to be trained
  - train_dataset: the training set of type NERDataset
  - batch_size: integer represents the number of examples per step
  - epochs: integer represents the total number of epochs (full training pass)
  - learning_rate: the learning rate to be used by the optimizer
  """
  # create the dataloader of the training set (make the shuffle=True)
  train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  # create the dataloader of the training set (make the shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
  # create a dictionary that holds both
  dataloader = {
      "train": train_dataloader,
      "val": val_dataloader}

  # GPU configuration
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(epochs):

      for phase in ['train', 'val']:
          if phase == 'train':  # put the model in training mode
              model.train()
          else:
              # put the model in validation mode, in order not to update parameters in dropout.
              model.eval()

          # keep track of training and validation loss    
          total_acc_train = 0
          total_loss_train = 0
          train_labels = []
          train_preds = []
        
          for train_input, train_attention_masks, train_label in dataloader[phase]:

              # move data to the device
              train_label = train_label.to(device)
              train_input = train_input.to(device)
              train_attention_masks = train_attention_masks.to(device)

              # do the forward pass
              output = model(train_input, train_attention_masks)
              
              # The docs of the loss function expects:
              # input of shape: (batch_size, n_classes)
              # target of shape: (btach_size)
              
              # loss calculation
              batch_loss = criterion(output, train_label)

              # append the batch loss to the total_loss_train
              total_loss_train += batch_loss
              
              # calculate the batch accuracy (just add the number of correct predictions)
              # torch.Tensor.item(): Returns the value of this tensor as a standard Python number. This only works for tensors with one element.
              train_pred = torch.argmax(output, dim=-1)

              num_correct_predictions = (train_pred == train_label).sum().item()
              acc = num_correct_predictions
              total_acc_train += acc

              if phase == 'train':
                  # zero your gradients
                  optimizer.zero_grad()
                  # do the backward pass
                  batch_loss.backward()
                  # update the weights with your optimizer
                  optimizer.step()
              
              # move data to cpu then numpy so you can make use of sklearn metric functions
              train_labels += list(train_label.to('cpu').detach().numpy())
              train_preds += list(train_pred.to('cpu').detach().numpy())
              
          # calculate epoch's loss
          # len(train_dataset) will call the __len__ of the NERDataset
          # will return the number of sentences in the dataset
          if phase == 'train':
            sentences_count = len(train_dataset)
          else:
            sentences_count = len(val_dataset)

          # Measuring performance
          # calculate epoch's accuracy and loss
          epoch_loss = total_loss_train / sentences_count
          epoch_acc = total_acc_train / sentences_count
          report = metrics.classification_report(train_labels, train_preds, target_names=classes_names, digits=4, output_dict=True)
          
          print(f'Epochs: {epoch_num + 1} | {phase} Loss: {epoch_loss} | {phase} Accuracy: {epoch_acc} | {phase} macro avg persision: {report["macro avg"]}\n')
          
          if epoch_num % 5==0:
            # calculate the classification report each 5 epochs
            report = metrics.classification_report(train_labels, train_preds, target_names=classes_names, digits=4)
            print(f'Classification Report:\n{report}\n')

def predict(model, test_dataset, classes_names, batch_size=16):
    #
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    n_classes = len(classes_names)
    # GPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    #
    model.eval()
    test_preds = []
    #
    for test_input, test_attention_masks in test_dataloader:
        # move data to the device
        test_input = test_input.to(device)
        test_attention_masks = test_attention_masks.to(device)
        # do the forward pass
        output = model(test_input, test_attention_masks)
        # torch.Tensor.item(): Returns the testue of this tensor as a standard Python number. This only works for tensors with one element.
        test_pred = torch.argmax(output, dim=-1)
        #
        test_preds += list(test_pred.to('cpu').detach().numpy())
    # 
    return test_preds

def eval_only(model, val_dataset, criterion, classes_names, batch_size=16):
    #
    val_labels = val_dataset.labels
    val_preds = predict(model, val_dataset, classes_names, batch_size)
    # 
    report = metrics.classification_report(val_labels, val_preds, target_names=classes_names, digits=4)
    print(f'Classification Report:\n{report}\n')
    return
