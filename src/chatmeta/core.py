from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class ChatMeta:
    def __init__(
        self,
        repo: str,
        subset: str = "default",
        split: str = "train",
        format: str | None = None,
        chat_column: str = "messages",
        speaker_column: str = "speaker",
        content_column: str = "content",
    ):
        ds = load_dataset(repo, subset, split=split)
        records = []
        for data in ds:
            record = []
            for m in data[chat_column]:
                record.append(
                    {
                        "role": m[speaker_column],
                        "content": m[content_column],
                    }
                )
            records.append({"messages": record, "id": data["id"]})
        self.dataset = Dataset.from_list(records)

        self.chat_column = chat_column

        if format is None:
            self.tokenizer = AutoTokenizer.from_pretrained("Spiral-AI/anonymous-7b")

    def __getitem__(
        self,
        idx: int = 0
    ):
        return self.tokenizer.apply_chat_template(
            self.dataset[self.chat_column][idx],
            tokenize=False,
            add_generation_prompt=False,
        )


    def print(
        self,
        idx: int = 0,
        add_generation_prompt: bool = False,
        chat_column: str | None = None,
    ):
        for m in self.dataset[self.chat_column or chat_column][idx]:
            print("-" * 80)
            print(
                self.tokenizer.apply_chat_template(
                    [m],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
