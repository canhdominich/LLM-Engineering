import torch
import torch.nn.functional as F
from torch import nn

PAD, BOS, EOS, VOCAB = 10, 11, 12, 13  # 0-9 là chữ số, thêm 3 ký hiệu đặc biệt


def make_batch(gen, size, min_len, max_len):
    lens = torch.randint(min_len, max_len + 1, (size,), generator=gen)
    src = torch.full((size, int(lens.max())), PAD)
    for i, n in enumerate(lens.tolist()):
        src[i, :n] = torch.randint(0, 10, (n,), generator=gen)
    tgt_in = torch.cat([torch.full((size, 1), BOS), src], dim=1)
    tgt_out = torch.cat([src, torch.full((size, 1), PAD)], dim=1)
    tgt_out[torch.arange(size), lens] = EOS
    return src, lens, tgt_in, tgt_out


class Seq2Seq(nn.Module):
    def __init__(self, hidden, attention):
        super().__init__()
        self.attention = attention
        self.emb = nn.Embedding(VOCAB, 32)
        self.encoder = nn.GRU(32, hidden, batch_first=True)
        self.decoder = nn.GRU(32, hidden, batch_first=True)
        self.out = nn.Linear(hidden * (2 if attention else 1), VOCAB)

    def encode(self, src, lens):
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), lens, batch_first=True, enforce_sorted=False)
        states, last = self.encoder(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True, total_length=src.shape[1])
        return states, last  # states: mọi bước (B, n, H); last: chỉ bước cuối (1, B, H)

    def predict(self, dec, states, src):
        if not self.attention:
            return self.out(dec), None
        scores = dec @ states.transpose(1, 2)                                   # (B, t, n)
        weights = scores.masked_fill((src == PAD)[:, None, :], float("-inf")).softmax(-1)
        context = weights @ states                                              # (B, t, H)
        return self.out(torch.cat([dec, context], dim=-1)), weights

    def forward(self, src, lens, tgt_in):
        states, last = self.encode(src, lens)
        dec, _ = self.decoder(self.emb(tgt_in), last)
        return self.predict(dec, states, src)[0]

    @torch.no_grad()
    def greedy(self, src, lens, steps):
        states, h = self.encode(src, lens)
        tok, outs, maps = torch.full((src.shape[0], 1), BOS), [], []
        for _ in range(steps):
            dec, h = self.decoder(self.emb(tok), h)
            logits, w = self.predict(dec, states, src)
            tok = logits.argmax(-1)
            outs.append(tok)
            maps.append(w)
        return torch.cat(outs, dim=1), maps


def train(hidden, attention, steps=1000, max_len=20):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(1)
    model = Seq2Seq(hidden, attention)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        src, lens, tgt_in, tgt_out = make_batch(gen, 64, 3, max_len)
        loss = F.cross_entropy(model(src, lens, tgt_in).reshape(-1, VOCAB), tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, loss.item()


def evaluate(model, length, size=300):
    src, lens, _, _ = make_batch(torch.Generator().manual_seed(100 + length), size, length, length)
    pred, _ = model.greedy(src, lens, length)
    right = pred == src
    return right.float().mean().item(), right.all(dim=1).float().mean().item()
