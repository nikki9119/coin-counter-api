import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def Detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        '', '', '', False, '', 640
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    weights = 'weights.pt'
    source = './INPUTS/image.jpg'
    out = './OUTPUTS'
    save_txt = './OUTPUTS'
    device = ''
    augment=False
    conf_thres = 0.5
    iou_thres = 0.6
    classes = None
    agnostic_nms = False

    set_logging()
    device = select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out) 
    os.makedirs(out) 
    half = device.type != 'cpu' 

    model = attempt_load(weights, map_location=device)  
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()  

    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']) 
        modelc.to(device).eval()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True 
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  
            if webcam:  
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    s += '%g %ss, ' % (n, names[int(c)])

                pred_lists = []
                for *xyxy, conf, cls in reversed(det):
                    flag = True
                    if save_txt: 
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        pred_str = ('%g ' * 5) % (cls, *xywh)
                        pred_list = pred_str.split(' ')
                        pred_list.pop()
                        pred_list.append('%.2f'%(conf))
                        for item in pred_lists:
                            if abs(float(item[1])-float(pred_list[1]))<0.05 and abs(float(item[2])-float(pred_list[2]))<0.05 and abs(float(item[3])-float(pred_list[3]))<0.05 and abs(float(item[4])-float(pred_list[4]))<0.05:
                                if(float(pred_list[5])<float(item[5])):
                                    flag=False
                                else:
                                    print(item)
                                    pred_lists.remove(item)
                        if flag:
                            pred_lists.append(pred_list)
                    if (save_img or view_img) and flag: 
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                file_list_preds = []
                for item in pred_lists:
                    pred_str = ' '.join(item)
                    file_list_preds.append(pred_str+'\n')
                with open(txt_path + '.txt','w') as fp:
                    fp.writelines(file_list_preds)

            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'): 
                    raise StopIteration

            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path: 
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release() 

                        fourcc = 'mp4v' 
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))