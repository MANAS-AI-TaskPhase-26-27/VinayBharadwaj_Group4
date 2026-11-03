import os
import urllib.request
from collections import deque
import cv2
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(FOLDER, "Volleyball.mp4")
OUTPUT = os.path.join(FOLDER, "tracked.mp4")

MODEL_CFG = os.path.join(FOLDER, "yolov4.cfg")
MODEL_WEIGHTS = os.path.join(FOLDER, "yolov4.weights")
MODEL_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
MODEL_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

PLAYER_CONFIDENCE = 0.25
COUNT_SAMPLES = 60
STATIC_RADIUS = 45
STATIC_SHARE = 0.7
PALETTE_SIZE = 6
COUNT_PERCENTILE = 90
MAX_TEAM_SIZE = 6

BALL_LOW = np.array([12, 90, 150])
BALL_HIGH = np.array([38, 255, 255])
NET_LINE_Y = 270
BALL_MIN_AREA, BALL_MAX_AREA = 40, 700

MAX_STEP = 90
MAX_GAP = 3
ARC_LENGTH = 8
ARC_MIN_TRAVEL = 70
ARC_MAX_ERROR = 2.0
TRAIL_LENGTH = 25

def find_candidates(video_path):
    video = cv2.VideoCapture(video_path)
    background_model = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=True)
    all_spots = []

    while True:
        ok, frame = video.read()
        if not ok: break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, BALL_LOW, BALL_HIGH)
        redder = cv2.inRange(hsv, np.array([3, BALL_LOW[1], BALL_LOW[2]]), BALL_HIGH)
        color[:NET_LINE_Y] = redder[:NET_LINE_Y]
        color = cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        moving = background_model.apply(frame)
        moving = cv2.threshold(moving, 200, 255, cv2.THRESH_BINARY)[1]
        moving[:50] = 0
        moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        count, labels, stats, centers = cv2.connectedComponentsWithStats(color)
        spots = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA): continue
            if max(w, h) > 90 or max(w, h) > 4 * min(w, h): continue
            blob = labels[y:y + h, x:x + w] == i
            if (moving[y:y + h, x:x + w] > 0)[blob].mean() < 0.08: continue
            spots.append((float(centers[i][0]), float(centers[i][1])))
        all_spots.append(spots)

    video.release()
    return all_spots

def follow_path(all_spots, start_frame, start_index):
    best_path = []
    first = np.array(all_spots[start_frame][start_index])

    for gap in range(1, MAX_GAP + 2):
        if start_frame + gap >= len(all_spots): break
        for second in all_spots[start_frame + gap]:
            step = np.array(second) - first
            if np.hypot(*step) > MAX_STEP * gap: continue
            path = [(start_frame, *first), (start_frame + gap, *second)]
            speed = step / gap
            frame, last = start_frame + gap, np.array(second)

            while True:
                extended = False
                for skip in range(1, MAX_GAP + 2):
                    nxt = frame + skip
                    if nxt >= len(all_spots): break
                    guess = last + speed * skip
                    near = [s for s in all_spots[nxt] if np.hypot(s[0] - guess[0], s[1] - guess[1]) < 45 + 4 * skip]
                    if near:
                        s = min(near, key=lambda s: np.hypot(s[0] - guess[0], s[1] - guess[1]))
                        speed = 0.5 * speed + 0.5 * (np.array(s) - last) / skip
                        last, frame = np.array(s), nxt
                        path.append((frame, *s))
                        extended = True
                        break
                if not extended: break

            if len(path) > len(best_path): best_path = path
    return best_path

def is_smooth_arc(points):
    frames = np.array([p[0] for p in points], float)
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    if np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < ARC_MIN_TRAVEL: return False
    
    line_x = np.polyfit(frames, xs, 1)
    curve_y = np.polyfit(frames, ys, 2)
    error = np.hypot(np.polyval(line_x, frames) - xs, np.polyval(curve_y, frames) - ys).mean()
    return error <= ARC_MAX_ERROR and abs(curve_y[0]) < 3

def find_ball_positions(all_spots):
    ball = {}
    used = set()

    for frame in range(len(all_spots)):
        for index in range(len(all_spots[frame])):
            if (frame, index) in used: continue
            path = follow_path(all_spots, frame, index)
            for p in path:
                for k, s in enumerate(all_spots[p[0]]):
                    if abs(s[0] - p[1]) < 0.01 and abs(s[1] - p[2]) < 0.01:
                        used.add((p[0], k))

            for start in range(0, len(path) - ARC_LENGTH + 1):
                window = path[start:start + ARC_LENGTH]
                if window[-1][0] - window[0][0] > ARC_LENGTH + 3: continue
                if is_smooth_arc(window):
                    for f, x, y in window: ball[f] = (x, y)

    for f in sorted(ball):
        for gap in range(2, MAX_GAP + 2):
            if f + gap in ball and all(f + k not in ball for k in range(1, gap)):
                for k in range(1, gap):
                    t = k / gap
                    ball[f + k] = ((1 - t) * ball[f][0] + t * ball[f + gap][0],
                                   (1 - t) * ball[f][1] + t * ball[f + gap][1])
    return ball

def download_model():
    for path, url in ((MODEL_CFG, MODEL_CFG_URL), (MODEL_WEIGHTS, MODEL_WEIGHTS_URL)):
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

def find_court(video_path):
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pictures = []
    for number in np.linspace(0, total - 1, 60).astype(int):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(number))
        ok, frame = video.read()
        if ok: pictures.append(frame)
    video.release()
    
    empty_court = np.median(np.stack(pictures), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(empty_court, cv2.COLOR_BGR2HSV)
    floor = cv2.inRange(hsv, np.array([3, 60, 120]), np.array([20, 255, 255]))
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(floor)
    biggest = 1 + np.argmax(stats[1:, 4])
    contours, _ = cv2.findContours((labels == biggest).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)

    top = outline[:, 1] < stats[biggest, 1] + 50
    outline[top, 1] -= 85
    outline[top & (outline[:, 0] < outline[:, 0].mean()), 0] -= 40
    outline[top & (outline[:, 0] >= outline[:, 0].mean()), 0] += 40
    return outline.reshape(-1, 1, 2).astype(np.int32), empty_court

def shirt_pixels(frame, still_picture, box):
    x, y, w, h = box
    x0, x1 = int(max(x + 0.2 * w, 0)), int(min(x + 0.8 * w, frame.shape[1] - 1))
    y0, y1 = int(max(y + 0.15 * h, 0)), int(min(y + 0.5 * h, frame.shape[0] - 1))
    if x1 <= x0 or y1 <= y0: return np.zeros((0, 3), np.float32)
    
    chest = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    empty = cv2.cvtColor(still_picture[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    differs = np.abs(chest - empty).sum(axis=2) > 55
    pixels = chest[differs]
    if len(pixels) > 120:
        pixels = pixels[np.random.RandomState(0).choice(len(pixels), 120, replace=False)]
    return pixels

def detect_people(video_path, court, empty_court, frame_numbers):
    download_model()
    net = cv2.dnn_DetectionModel(MODEL_CFG, MODEL_WEIGHTS)
    net.setInputParams(scale=1 / 255.0, size=(608, 608), swapRB=True)
    video = cv2.VideoCapture(video_path)
    people = {}
    
    for number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(number))
        ok, frame = video.read()
        if not ok: continue
        ids, scores, boxes = net.detect(frame, confThreshold=PLAYER_CONFIDENCE, nmsThreshold=0.4)
        found = []
        for class_id, box in zip(np.array(ids).flatten(), boxes):
            if class_id != 0: continue
            x, y, w, h = [int(v) for v in box]
            if h > 400 or w > 300 or h < 30: continue
            feet = (float(x + w / 2), float(min(y + h, frame.shape[0] - 20)))
            if cv2.pointPolygonTest(court, feet, True) < -10: continue
            found.append(((x, y, w, h), shirt_pixels(frame, empty_court, (x, y, w, h))))
        people[int(number)] = found
    video.release()
    return people

def remove_standing_still(people):
    feet = {f: np.array([[b[0] + b[2] / 2, b[1] + b[3]] for b, _ in people[f]]).reshape(-1, 2) for f in people}
    kept = {}
    for f in people:
        kept[f] = []
        for (box, pixels), (fx, fy) in zip(people[f], feet[f]):
            seen = [len(feet[g]) > 0 and (np.hypot(feet[g][:, 0] - fx, feet[g][:, 1] - fy) < STATIC_RADIUS).any() for g in people]
            if np.mean(seen) < STATIC_SHARE:
                kept[f].append((box, pixels))
    return kept

def colour_name(lab_pixels):
    pixels = np.clip(lab_pixels, 0, 255).astype(np.uint8).reshape(1, -1, 3)
    hsv = cv2.cvtColor(cv2.cvtColor(pixels, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = [int(c) for c in np.median(hsv, axis=0)]
    if v < 60: return "black"
    if s < 40: return "white" if v > 180 else "grey"
    if h < 6 or h >= 165: return "red"
    if h < 13: return "orange"
    if h < 35: return "yellow"
    if h < 85: return "green"
    if h < 135: return "blue"
    return "purple"

def split_into_teams(people):
    stop = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    everyone = [(f, i) for f in people for i in range(len(people[f])) if len(people[f][i][1]) >= 15]
    all_pixels = np.vstack([people[f][i][1] for f, i in everyone]).astype(np.float32)
    _, _, palette = cv2.kmeans(all_pixels, PALETTE_SIZE, None, stop, 3, cv2.KMEANS_PP_CENTERS)

    def shirt_profile(pixels):
        nearest = ((pixels[:, None, :] - palette[None]) ** 2).sum(axis=2).argmin(axis=1)
        share = np.bincount(nearest, minlength=PALETTE_SIZE).astype(np.float32)
        return share / share.sum()

    profiles = np.array([shirt_profile(people[f][i][1]) for f, i in everyone], np.float32)
    _, group, centres = cv2.kmeans(profiles, 2, None, stop, 10, cv2.KMEANS_PP_CENTERS)
    team_of_person = {person: int(g) for person, g in zip(everyone, group.flatten())}
    names = [colour_name(np.vstack([people[f][i][1] for (f, i), t in team_of_person.items() if t == team])) for team in range(2)]
    if names[0] == names[1]: names = [names[0] + " 1", names[1] + " 2"]
    return team_of_person, names

def count_players(video_path, court, empty_court):
    total = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    frame_numbers = np.linspace(0, total - 1, COUNT_SAMPLES).astype(int)
    people = detect_people(video_path, court, empty_court, frame_numbers)
    people = remove_standing_still(people)
    team_of_person, names = split_into_teams(people)
    
    per_frame = {f: [0, 0] for f in people}
    for (f, i), team in team_of_person.items():
        per_frame[f][team] += 1
        
    counts = []
    for team in range(2):
        seen = [per_frame[f][team] for f in per_frame]
        counts.append(min(MAX_TEAM_SIZE, int(round(np.percentile(seen, COUNT_PERCENTILE)))))
    return names, counts

def draw_video(ball, names, counts):
    video = cv2.VideoCapture(VIDEO)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail = deque(maxlen=TRAIL_LENGTH)
    frame_number = 0

    while True:
        ok, frame = video.read()
        if not ok: break

        if frame_number in ball:
            x, y = int(ball[frame_number][0]), int(ball[frame_number][1])
            trail.append((x, y))
            cv2.circle(frame, (x, y), 22, (0, 255, 0), 2)
        elif frame_number - 1 not in ball:
            trail.clear()

        for i in range(1, len(trail)):
            fade = i / len(trail)
            cv2.line(frame, trail[i - 1], trail[i], (0, int(255 * fade), 255), max(1, int(4 * fade)))

        cv2.rectangle(frame, (0, height - 40), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, f"{names[0].upper()} TEAM: {counts[0]} players   {names[1].upper()} TEAM: {counts[1]} players",
                    (15, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Volleyball Vision", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frame_number += 1

    video.release()
    writer.release()
    cv2.destroyAllWindows()

def main():
    if not os.path.exists(VIDEO): 
        print(f"Cannot find video: {VIDEO}")
        return
    
    print("Step 1/4: Looking for possible ball spots (this takes about a minute)...")
    all_spots = find_candidates(VIDEO)
    
    print("Step 2/4: Finding the ball's flight paths...")
    ball = find_ball_positions(all_spots)
    print(f"Ball found in {len(ball)} of {len(all_spots)} frames")
    
    print("Step 3/4: Finding the teams and counting their players (this takes several minutes)...")
    court, empty_court = find_court(VIDEO)
    names, counts = count_players(VIDEO, court, empty_court)
    print(f"Players  {names[0]} team: {counts[0]}   {names[1]} team: {counts[1]}")
    
    print("Step 4/4: Drawing the result...")
    draw_video(ball, names, counts)
    print("Done! Saved as", OUTPUT)

if __name__ == "__main__":
    main()

