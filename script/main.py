from processing import *
from tokenizer import *
from dataset import *
from trainer import *
from config import *

train_data = read_data_from_dir(TRAIN_DIR)
valid_data = read_data_from_dir(VALID_DIR)
test_data = read_data_from_dir(TEST_DIR)

train_data = preprocessing(train_data)
valid_data = preprocessing(valid_data)
test_data = preprocessing(test_data)

vocab, tokenizer = build_tokenizer_vocab(train_data, VOCAB_SAVE_PATH, VOCAB_SIZE)

train_dataset = TextClassificationDataset(train_data, vocab, tokenizer)
valid_dataset = TextClassificationDataset(valid_data, vocab, tokenizer)
test_dataset = TextClassificationDataset(test_data, vocab, tokenizer)

train_dataloader = train_dataset.get_dataloader(BATCH_SIZE, True, DEVICE)
valid_dataloader = valid_dataset.get_dataloader(BATCH_SIZE, False, DEVICE)
test_dataloader = test_dataset.get_dataloader(BATCH_SIZE, False, DEVICE)

trainer = Trainer(MODEL_SAVE_PATH, SAVE_EPOCH, LOG_INTERVAL, DEVICE)
VOCAB_SIZE = len(vocab)
model = trainer.prepare_model(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASS)
trainer.configure_optimizers(model)
model = trainer.train(20, model, train_dataloader)
trainer.evaluate(model, valid_dataloader)

