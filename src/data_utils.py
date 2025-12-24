class DataManager:
    def __init__(self, url: str, max_chars: int = 50000):
        self.url = url
        self.max_chars = max_chars
        self.text = self._fetch_text()

    def _fetch_text(self) -> str:
        response = requests.get(self.url, timeout=15)
        return response.text[:self.max_chars]

    def get_dataloader(self, seq_length: int, batch_size: int):
        dataset = TextDataset(self.text, seq_length)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        ), dataset