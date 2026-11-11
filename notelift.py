import argparse, json
from pathlib import Path
import cv2, numpy as np, torch, torchvision
import torch.nn as nn, torch.nn.functional as F

OUT, MODELS = Path("outputs"), Path("models")
for d in (Path("data"), OUT, MODELS):
    d.mkdir(exist_ok=True)

SIZE = 256                  # synthetic images are SIZE x SIZE
CORNER_RES = 224            # what ResNet expects
LABEL_SPACE = (800, 600)    # label.py clicks on an 800x600 preview
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STREAMS = {"train": 0, "val": 900_000, "test": 2_000_000}   # these never overlap

# ------------------------------------------------------------- training data

def glyphs():
    """Handwritten characters from EMNIST - the ink in every fake photo."""
    ds = torchvision.datasets.EMNIST("data", split="balanced", train=True, download=True)
    return ds.data.numpy().transpose(0, 2, 1)       # EMNIST ships rotated + mirrored

def stamp(ink, shape, mark, alpha):
    """Record a mark twice: how dark the camera sees it, and that it IS writing,
    so the right answer never depends on how faintly it happened to be drawn."""
    np.maximum(ink, mark * alpha, out=ink)
    np.maximum(shape, mark, out=shape)

def write_text(chars, rng, ink, shape, n=SIZE):
    """Lines of characters, from tiny axis labels up to a heading."""
    y = int(rng.integers(5, 30))
    while y < n - 20:
        h = int(rng.integers(8, 60))
        if y + h >= n:
            break
        line, x = np.zeros((n, n), np.float32), int(rng.integers(5, 40))
        while x < n - h:
            w = max(4, int(h * rng.uniform(0.7, 1.0)))
            if x + w >= n:
                break
            line[y:y+h, x:x+w] = np.maximum(line[y:y+h, x:x+w],
                                            cv2.resize(chars[rng.integers(0, len(chars))], (w, h)) / 255)
            x += w + int(rng.integers(2, 12))
        stamp(ink, shape, line, float(rng.uniform(0.35, 1.0)))      # its own darkness
        y += h + int(rng.integers(6, 28))

def draw_marks(rng, ink, shape, n=SIZE):
    """Rules, axes, curves, boxes and circles - the parts of a board that are not
    letters. Without them the model decides writing means small blobs, and drops
    the gridlines of a graph. Thickness reaches 1px because that is the hard case."""
    spot = lambda: (int(rng.integers(0, n)), int(rng.integers(0, n)))
    for _ in range(int(rng.integers(1, 6))):
        m, t, kind = np.zeros((n, n), np.float32), int(rng.integers(1, 4)), rng.integers(5)
        if kind == 0:                                   # a rule, half of them dashed
            y, x = int(rng.integers(0, n)), int(rng.integers(0, n // 2))
            dash, gap = (6, 12) if rng.random() < 0.5 else (n, n)
            for s in range(x, n, gap):
                cv2.line(m, (s, y), (min(s + dash, n), y), 1.0, t)
        elif kind == 1:                                 # axes with tick marks
            x, y = int(rng.integers(10, n // 2)), int(rng.integers(n // 2, n - 10))
            cv2.line(m, (x, y), (x, int(rng.integers(5, max(6, y - 30)))), 1.0, t)
            cv2.line(m, (x, y), (int(rng.integers(x + 50, n)), y), 1.0, t)
            for i in range(1, 8):
                cv2.line(m, (x + i*14, y - 3), (x + i*14, y + 3), 1.0, 1)
        elif kind == 2:                                 # a plotted curve
            xs = np.linspace(0, n, int(rng.integers(4, 10)))
            ys = rng.integers(n // 8, n, len(xs))
            cv2.polylines(m, [np.stack([xs, ys], 1).astype(np.int32)], False, 1.0, t)
        elif kind == 3:                                 # a box, maybe a table
            x, y = spot()
            w, h = int(rng.integers(40, n // 2)), int(rng.integers(30, n // 2))
            cv2.rectangle(m, (x, y), (x + w, y + h), 1.0, t)
            for i in range(1, int(rng.integers(1, 5))):
                cv2.line(m, (x, y + i*h//5), (x + w, y + i*h//5), 1.0, 1)
        else:
            cv2.ellipse(m, spot(), (int(rng.integers(12, n//4)), int(rng.integers(8, n//4))),
                        float(rng.uniform(0, 180)), 0, 360, 1.0, t)
        stamp(ink, shape, m, float(rng.uniform(0.25, 1.0)))

def content(chars, rng, n=SIZE):
    """Mostly writing, and more often than not a diagram sitting on top of it."""
    ink, shape = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
    if rng.random() < 0.9:
        write_text(chars, rng, ink, shape, n)
    if rng.random() < 0.6 or shape.max() == 0:
        draw_marks(rng, ink, shape, n)
    return ink, shape

def soft_field(n, rng, lo, hi):
    """A 4x4 grid of noise blown up smooth - stands in for uneven lighting."""
    f = cv2.resize(rng.random((4, 4)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    return lo + (hi - lo) * (f - f.min()) / (np.ptp(f) + 1e-6)

def fake_photo(ink, ghost, rng, severity, dark):
    """Photograph `ink`. We drew it, so the right answer costs nothing."""
    n = ink.shape[0]
    blur = cv2.GaussianBlur(ghost, (0, 0), rng.uniform(2, 5))    # last class, half erased
    if dark:                                                    # chalk on a blackboard
        img = soft_field(n, rng, 0.05, 0.30) + rng.uniform(0.04, 0.13) * blur
        img += rng.uniform(0.45, 0.75) * ink * soft_field(n, rng, 0.45, 1.0)
    else:                                                       # ink on paper or a slide
        img = soft_field(n, rng, 0.75, 0.98) - rng.uniform(0.02, 0.08) * blur
        img -= rng.uniform(0.45, 0.72) * ink * soft_field(n, rng, 0.70, 1.0)
    img *= soft_field(n, rng, 1 - 0.5 * severity, 1.0)          # a shadow across it
    glare = np.zeros((n, n), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cv2.ellipse(glare, (int(rng.integers(0, n)), int(rng.integers(0, n))),
                    (int(rng.integers(n//8, n//3)), int(rng.integers(n//12, n//4))),
                    float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    img += cv2.GaussianBlur(glare, (0, 0), n / 12) * (0.15 + 0.55 * severity)
    img = cv2.GaussianBlur(img, (0, 0), 0.4 + 1.2 * severity)   # out of focus
    img += rng.normal(0, 0.01 + 0.03 * severity, img.shape)     # sensor noise
    return np.clip(img, 0, 1).astype(np.float32)

def sample(chars, rng, severity=None, dark=None):
    """One (photo, answer, surface). severity 0 = clean, 1 = awful."""
    severity = rng.uniform(0, 1) if severity is None else severity
    dark = rng.random() < 0.5 if dark is None else dark
    ink, shape = content(chars, rng)
    photo = fake_photo(ink, content(chars, rng)[1], rng, severity, dark)
    return photo, (shape > 0.5).astype(np.float32), dark

def fixed_set(chars, split, n):
    """A reproducible set from one stream, spread over both surfaces."""
    return [sample(chars, np.random.default_rng(STREAMS[split] + i), (i % 5) / 4, i % 2 == 0)
            for i in range(n)]

# ----------------------------------------------------- classical comparisons

def iou(pred, truth):
    """Overlap on writing pixels. Writing is ~5% of the image, so plain accuracy
    is useless - answering 'blank everywhere' would score 95%."""
    a, b = pred > 0.5, truth > 0.5
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union

def otsu(photo):
    """One brightness cutoff for the whole image."""
    g = (photo * 255).astype(np.uint8)
    return (cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 127).astype(np.float32)

def adaptive(photo, dark=True, block=61, c=28):
    """Compare each pixel to its neighbourhood. Chalk is brighter than its
    surroundings and slide text darker, so the test flips with the surface."""
    g = (photo * 255).astype(np.uint8)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY if dark else cv2.THRESH_BINARY_INV,
                               block, -c if dark else c)
    return (th > 127).astype(np.float32)

def is_board(gray):
    """Dark board or bright slide? Writing covers a few percent, so the typical
    pixel is the surface itself. Median, because glare drags a mean around."""
    return bool(np.median(gray) < 0.5)

# ---------------------------------------------------------------- the models

def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(True))

class UNet(nn.Module):
    """Squeeze the image down to understand it, build it back up to locate it.
    The skip connections stop 3-pixel strokes vanishing on the way through."""
    def __init__(self, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.down1, self.down2, self.down3 = block(1, base), block(base, base*2), block(base*2, base*4)
        self.middle = block(base*4, base*8)
        self.up3, self.up2, self.up1 = block(base*12, base*4), block(base*6, base*2), block(base*3, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        a = self.down1(x)
        b = self.down2(self.pool(a))
        c = self.down3(self.pool(b))
        x = self.middle(self.pool(c))
        for up, skip in ((self.up3, c), (self.up2, b), (self.up1, a)):
            x = up(torch.cat([F.interpolate(x, scale_factor=2), skip], 1))
        return self.head(x)

def corner_model(pretrained=False):
    """ResNet-18 with its 1000-way classifier swapped for 8 numbers: four corners
    as (x, y) fractions. Its ImageNet weights already know about edges and
    rectangles, which is the only reason 44 photos is enough to learn from."""
    w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    net = torchvision.models.resnet18(weights=w)
    net.fc = nn.Sequential(nn.Linear(net.fc.in_features, 8), nn.Sigmoid())
    return net

def loss_fn(logits, truth):
    """Cross-entropy learns smoothly but is happy to answer 'blank' when writing
    is 5% of pixels. Dice scores overlap, so blank scores zero. Use both."""
    p = torch.sigmoid(logits)
    hit = (p * truth).sum((1, 2, 3))
    dice = 1 - (2*hit + 1) / (p.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, truth) + dice.mean()

def load(name, build):
    """Load a saved model, or None if it has not been trained yet."""
    if not (MODELS / name).exists():
        return None
    net = build().to(DEVICE)
    net.load_state_dict(torch.load(MODELS / name, map_location=DEVICE))
    return net.eval()

@torch.no_grad()
def predict(net, photo):
    """Where is the writing? One probability per pixel, thresholded at a half."""
    h, w = photo.shape
    pad = np.pad(photo, ((0, -h % 8), (0, -w % 8)), mode="edge")    # pooling halves 3x
    p = torch.sigmoid(net(torch.from_numpy(pad)[None, None].to(DEVICE)))
    return (p[0, 0].cpu().numpy()[:h, :w] > 0.5).astype(np.float32)

def to_tensor(bgr):
    x = cv2.resize(bgr, (CORNER_RES, CORNER_RES)).astype(np.float32) / 255
    return torch.from_numpy(x.transpose(2, 0, 1))

# ---------------------------------------------------------- cleaning a photo

@torch.no_grad()
def rectify(img, net, margin=1.02):
    """Find the board's four corners and warp it flat."""
    h, w = img.shape[:2]
    pts = net(to_tensor(img)[None].to(DEVICE))[0].cpu().numpy().reshape(4, 2) * [w, h]
    # predictions sit slightly inside the board, and clipping text is worse than
    # keeping a sliver of wall, so push the corners outwards a little
    pts = (pts.mean(0) + (pts - pts.mean(0)) * margin).astype(np.float32)
    tl, tr, br, bl = pts
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 32 or H < 32:
        return img, False
    target = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, target), (W, H)), True

def tidy(mask, min_px=30, edge=0.015):
    """Drop what cannot be writing: specks too small to be a stroke, and long
    blobs lying along the frame, which are the screen's bezel."""
    h, w = mask.shape
    pad = int(edge * min(h, w))
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask * 255).astype(np.uint8), 8)
    keep = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        edgy = x <= pad or y <= pad or x + bw >= w - pad or y + bh >= h - pad
        if area >= min_px and not (edgy and (bw > w/2 or bh > h/2 or min(bw, bh) <= 3)):
            keep.append(i)
    out = np.isin(lab, keep).astype(np.float32)
    out[:pad], out[-pad:], out[:, :pad], out[:, -pad:] = 0, 0, 0, 0
    return out

def straighten(mask, gray, limit=12):
    """Writing runs in horizontal lines, so the row sums are spikiest when the
    page is level. Try small rotations, keep the spikiest, turn both images by
    it. This removes the few degrees of tilt the corner finder leaves behind."""
    s = cv2.resize(mask, (0, 0), fx=300/mask.shape[0], fy=300/mask.shape[0])
    angles = np.arange(-limit, limit + 0.5, 0.5)
    spiky = [cv2.warpAffine(s, cv2.getRotationMatrix2D((s.shape[1]/2, s.shape[0]/2), a, 1.0),
                            s.shape[::-1]).sum(1).var() for a in angles]
    angle = float(angles[int(np.argmax(spiky))])
    turn = lambda im, fill: cv2.warpAffine(
        im, cv2.getRotationMatrix2D((im.shape[1]/2, im.shape[0]/2), angle, 1.0),
        im.shape[1::-1], borderValue=fill)
    return turn(mask, 0.0), turn(gray, float(np.median(gray))), angle

def enhance(gray, mask, dark):
    """Look like a scan, not a fax. A hard black-or-white mask snaps thin strokes
    into gravel, so keep the real grey tones where the model found writing and
    flatten the rest to white. Closing with a big kernel erases the strokes and
    leaves the lighting; dividing by that cancels shadow and glare at once."""
    g = 1.0 - gray if dark else gray                # chalk is bright, so flip it
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    lighting = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, k), (0, 0), 20)
    flat = np.clip(g / np.maximum(lighting, 0.05), 0, 1)
    ink = flat[mask > 0.5]
    black = float(np.percentile(ink, 25)) if ink.size > 50 else 0.45
    levels = np.clip((flat - black) / (1 - black + 1e-6), 0, 1)
    reach = cv2.GaussianBlur(cv2.dilate(mask, np.ones((5, 5), np.uint8)), (0, 0), 2)
    return 1 - (1 - levels) * reach                 # outside the writing, pure white

# -------------------------------------------------------------------figures

def chart(x, series, labels, path, xlabel, ylabel, title):
    """A line chart - the accuracy comparison and the training curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, label in zip(series, labels):
        ax.plot(x, y, "o-" if len(x) < 20 else "-", linewidth=2, label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)

def sheet(images, path, cols=6, tile=(320, 240)):
    """Lay images out in a grid and save the lot as one picture."""
    out = []
    for im in images:
        if im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        out.append(cv2.resize(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), tile))
    out += [np.full((tile[1], tile[0], 3), 255, np.uint8)] * (-len(out) % cols)
    cv2.imwrite(str(path), np.vstack([np.hstack(out[i:i+cols]) for i in range(0, len(out), cols)]))
    print("saved", path)

# ------------------------------------------------------------------ commands

def preview(args):
    chars, rng = glyphs(), np.random.default_rng(SEED)
    sheet([sample(chars, rng, s, d)[0] for d in (True, False) for s in np.linspace(0, 1, 6)],
          OUT / "synthetic.png")

def train(args):
    print("device:", DEVICE)
    torch.manual_seed(SEED)
    chars = glyphs()
    val = fixed_set(chars, "val", 40)
    vx = torch.from_numpy(np.stack([p for p, _, _ in val]))[:, None].to(DEVICE)
    vy = np.stack([m for _, m, _ in val])
    net = UNet().to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(STREAMS["train"])
    best, log = 0.0, []
    for step in range(1, args.steps + 1):
        batch = [sample(chars, rng) for _ in range(args.batch)]
        x = torch.from_numpy(np.stack([p for p, _, _ in batch]))[:, None].to(DEVICE)
        y = torch.from_numpy(np.stack([m for _, m, _ in batch]))[:, None].to(DEVICE)
        loss = loss_fn(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % args.every:
            continue
        net.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(net(vx)) > 0.5).float().cpu().numpy()[:, 0]
        net.train()
        score = float(np.mean([iou(p, t) for p, t in zip(pred, vy)]))
        log.append((step, loss.item(), score))
        mark = ""
        if score > best:                                # keep the best, not the last
            best, mark = score, "  *saved"
            torch.save(net.state_dict(), MODELS / "unet.pt")
        print(f"step {step:>5}  loss {loss.item():.4f}  val IoU {score:.4f}{mark}")
    log = np.array(log)
    np.savetxt(OUT / "train_log.csv", log, delimiter=",", header="step,loss,val_iou", comments="")
    chart(log[:, 0], [log[:, 1], log[:, 2]], ["training loss", "validation IoU"],
          OUT / "training.png", "step", "", "Learning to pick writing out of a photo")
    print(f"best val IoU {best:.4f}  ->  {MODELS / 'unet.pt'}")

def corner_split(folder, seed=SEED):
    """Load the hand-labelled photos and split them. Corners were clicked on an
    800x600 preview, so that - not the original file size - is what they divide
    by. The extra document photos are off-domain: they help training, never scoring."""
    def read(sub):
        images, targets, names = [], [], []
        for name, pts in sorted(json.load(open(Path(folder) / sub / "labels.json")).items()):
            img = cv2.imread(str(Path(folder) / sub / name))
            p = (np.array(pts, np.float32) / LABEL_SPACE).astype(np.float32)
            if img is None or cv2.contourArea(p) < 0.02:
                continue
            images.append(cv2.resize(img, (CORNER_RES, CORNER_RES)))
            targets.append(p.ravel())
            names.append(name)
        return images, np.stack(targets).astype(np.float32), names

    images, targets, names = read("lecture")
    order = np.random.default_rng(seed).permutation(len(images))
    n = max(4, len(images) // 7)
    sets = {k: ([images[i] for i in v], targets[v], [names[i] for i in v])
            for k, v in (("test", order[:n]), ("val", order[n:2*n]), ("train", order[2*n:]))}
    if (Path(folder) / "extra").exists():
        ei, et, en = read("extra")
        ti, tt, tn = sets["train"]
        sets["train"] = (ti + ei, np.concatenate([tt, et]), tn + en)
    return sets

def jitter(img, pts, rng):
    """Spin, zoom and shift the photo, moving the corners with it. With only a few
    dozen photos this is most of what stops the model memorising them."""
    M = cv2.getRotationMatrix2D((CORNER_RES/2,)*2, rng.uniform(-15, 15), rng.uniform(0.85, 1.15))
    M[:, 2] += rng.uniform(-12, 12, 2)
    out = cv2.warpAffine(img, M, (CORNER_RES,)*2, borderMode=cv2.BORDER_REPLICATE)
    p = (M[:, :2] @ (pts.reshape(4, 2) * CORNER_RES).T).T + M[:, 2]
    return out, (p / CORNER_RES).ravel().astype(np.float32)

def corner_error(net, images, targets):
    """How far a predicted corner lands from the real one, as a fraction of the
    image width. 0.05 means five percent out."""
    with torch.no_grad():
        pred = net(torch.stack([to_tensor(i) for i in images]).to(DEVICE)).cpu()
    return (pred - torch.from_numpy(targets)).view(-1, 4, 2).norm(dim=2).mean().item()

def train_corners(args):
    sets = corner_split(args.data)
    rng = np.random.default_rng(SEED)
    print("  ".join(f"{k} {len(v[0])}" for k, v in sets.items()))
    json.dump({k: sets[k][2] for k in ("val", "test")}, open(MODELS / "corner_split.json", "w"))
    images, targets, _ = sets["train"]
    net = corner_model(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = 9e9
    for epoch in range(1, args.epochs + 1):
        net.train()
        for chunk in np.array_split(rng.permutation(len(images)), max(1, len(images) // 16)):
            pairs = [jitter(images[i], targets[i], rng) for i in chunk]
            x = torch.stack([to_tensor(im) for im, _ in pairs])
            x = (x * rng.uniform(0.7, 1.3) + rng.uniform(-0.15, 0.15)).clamp(0, 1)
            y = torch.from_numpy(np.stack([p for _, p in pairs]))
            loss = F.smooth_l1_loss(net(x.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if epoch % 5:
            continue
        net.eval()
        err = corner_error(net, *sets["val"][:2])
        mark = ""
        if err < best:
            best, mark = err, "  *saved"
            torch.save(net.state_dict(), MODELS / "corners.pt")
        print(f"epoch {epoch:>3}  val corner error {err*100:.1f}%{mark}")
    print(f"\nbest val {best*100:.1f}%   held-out test "
          f"{corner_error(load('corners.pt', corner_model), *sets['test'][:2])*100:.1f}%")

def evaluate(args):
    """The exam. Nothing here was used to train or to pick a model."""
    net = load("unet.pt", UNet)
    if net is None:
        raise SystemExit("No trained model. Run: python notelift.py train")
    chars = glyphs()
    tests = fixed_set(chars, "test", args.n)
    methods = (lambda p, d: otsu(p), adaptive, lambda p, d: predict(net, p))
    print(f"{'severity':>9}{'Otsu':>9}{'Adaptive':>10}{'U-Net':>9}")
    rows = []
    for k in range(5):                              # fixed_set lays severity out as i % 5
        group = tests[k::5]
        rows.append([k/4] + [float(np.mean([iou(f(p, d), m) for p, m, d in group])) for f in methods])
        print(f"{rows[-1][0]:>9.2f}{rows[-1][1]:>9.3f}{rows[-1][2]:>10.3f}{rows[-1][3]:>9.3f}")
    rows = np.array(rows)
    print(f"{'mean':>9}{rows[:,1].mean():>9.3f}{rows[:,2].mean():>10.3f}{rows[:,3].mean():>9.3f}")
    for label, want in (("board", True), ("slide", False)):
        g = [(p, m) for p, m, d in tests if d == want]
        print(f"  U-Net on {label}s only: {np.mean([iou(predict(net, p), m) for p, m in g]):.3f}")
    print(f"surface detection: {np.mean([is_board(p) == d for p, _, d in tests])*100:.1f}% correct")
    corners = load("corners.pt", corner_model)
    if corners is not None and Path(args.data).exists():
        test = corner_split(args.data)["test"]
        print(f"corner finder on {len(test[0])} held-out photos: "
              f"{corner_error(corners, *test[:2])*100:.1f}% error")
    np.savetxt(OUT / "results.csv", rows, delimiter=",", fmt="%.4f",
               header="severity,otsu,adaptive,unet", comments="")
    chart(rows[:, 0], [rows[:, 1], rows[:, 2], rows[:, 3]],
          ["Otsu", "Adaptive threshold", "U-Net (ours)"], OUT / "results.png",
          "degradation severity", "IoU on writing pixels",
          "Writing extraction accuracy vs photo quality")
    photo, mask, dark = tests[4]
    sheet([photo, mask, otsu(photo), adaptive(photo, dark), predict(net, photo)],
          OUT / "comparison.png", cols=5)

def clean_up(net, corners, path, args):
    """The whole pipeline on one photo: find the board, flatten it, read it."""
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read {path}")
    flat, found = (img, False) if args.no_rectify else rectify(img, corners, args.margin)
    gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (int(w * args.size / h), args.size)).astype(np.float32) / 255
    board = is_board(gray) if args.mode == "auto" else args.mode == "board"
    mask, gray, angle = straighten(tidy(predict(net, gray), args.min_px), gray)
    clean = enhance(gray, mask, board)
    warn = "  <- board too small in frame, strokes are lost" if h < 400 else ""
    print(f"{path.name:<14} {'cropped' if found else 'full':<8} {'board' if board else 'slide':<6}"
          f" {angle:>+5.1f} deg {mask.mean()*100:>5.1f}% writing{warn}")
    return clean, [img, flat, adaptive(gray, board), clean], found

def run(args):
    net, corners = load("unet.pt", UNet), load("corners.pt", corner_model)
    if net is None:
        raise SystemExit("No trained model. Run: python notelift.py train")
    if corners is None and not args.no_rectify:
        raise SystemExit("No corner model. Run: python notelift.py train-corners")
    target = Path(args.image)
    photos = [target] if target.is_file() else sorted(
        p for p in target.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        raise SystemExit(f"No images in {target}")
    pages, hits = [], 0
    for p in photos:
        clean, panels, found = clean_up(net, corners, p, args)
        hits += found
        if len(photos) == 1:
            cv2.imwrite(str(OUT / "clean.png"), (clean * 255).astype(np.uint8))
            return sheet(panels, OUT / "demo.png", cols=4, tile=(420, 320))
        (OUT / "batch").mkdir(exist_ok=True)
        cv2.imwrite(str(OUT / "batch" / f"{p.stem}.png"), (clean * 255).astype(np.uint8))
        pages.append(clean)
    print(f"\n{len(photos)} photos, cropped {hits} ({100*hits/len(photos):.0f}%)")
    sheet(pages, OUT / "batch_sheet.png")

def export(args):
    """Save both models as ONNX so the Android app can run them on the phone."""
    import onnx
    for name, net, dummy, axes in (
            ("notelift.onnx", load("unet.pt", UNet), torch.randn(1, 1, 768, 1024),
             {"image": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}}),
            ("corners.onnx", load("corners.pt", corner_model),
             torch.randn(1, 3, CORNER_RES, CORNER_RES), None)):
        if net is None:
            print("skipping", name, "- not trained yet")
            continue
        torch.onnx.export(net.cpu(), dummy, str(MODELS / name), input_names=["image"],
                          output_names=["logits"], dynamic_axes=axes, opset_version=18)
        onnx.save_model(onnx.load(str(MODELS / name)), str(MODELS / name), save_as_external_data=False)
        Path(str(MODELS / name) + ".data").unlink(missing_ok=True)      # keep it one file
        print(f"exported {MODELS / name}  {(MODELS / name).stat().st_size/1048576:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preview"), sub.add_parser("export")
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=2500)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--every", type=int, default=100)
    c = sub.add_parser("train-corners")
    c.add_argument("--data", default="corner_data")
    c.add_argument("--epochs", type=int, default=60)
    e = sub.add_parser("evaluate")
    e.add_argument("--n", type=int, default=100)
    e.add_argument("--data", default="corner_data")
    r = sub.add_parser("run")
    r.add_argument("--image", required=True, help="a photo, or a folder of them")
    r.add_argument("--mode", choices=["auto", "board", "slide"], default="auto")
    r.add_argument("--size", type=int, default=1000)
    r.add_argument("--min-px", type=int, default=30)
    r.add_argument("--margin", type=float, default=1.02)
    r.add_argument("--no-rectify", action="store_true")
    args = parser.parse_args()
    dict(preview=preview, train=train, evaluate=evaluate, run=run, export=export,
         **{"train-corners": train_corners})[args.command](args)

if __name__ == "__main__":
    main()