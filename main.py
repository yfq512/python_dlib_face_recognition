import dlib
import cv2
import numpy as np


def img2mask2(img, p1, p2):
    if p1[0] == p2[0]:
        p2[0] = p2[0] + 0.0001
    shape = img.shape
    mask = np.ones((shape[0], shape[1]))
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - k * p2[0]
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            zeros_jz_y = k * i + b
            if j > zeros_jz_y:
                mask[j, i] = 0

    (bb, gg, rr) = cv2.split(img)
    bb = np.multiply(bb, mask)
    gg = np.multiply(gg, mask)
    rr = np.multiply(rr, mask)
    result = cv2.merge((bb, gg, rr))
    return result.astype('uint8')


def img2mask(img, p1, p2):
    if p1[0] == p2[0]:
        p2[0] = p2[0] + 0.0001
    shape = img.shape
    mask = np.ones(shape)
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - k * p2[0]
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            zeros_jz_y = k * i + b
            if j > zeros_jz_y:
                mask[j, i] = [0, 0, 0]

    result = np.multiply(img, mask)
    return result.astype('uint8')


def main(img_path='./1.jpg'):
    detector = dlib.get_frontal_face_detector()  # 人脸box检测器
    image = cv2.imread(img_path)
    # image = dlib.load_rgb_image(img_path)
    # image = image[:, :, ::-1]
    res = detector(image, 2)
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 人脸关键点提取（此处是68点，也可以改为5点）
    detected_landmarks = predictor(image, res[0])
    feature = dlib.face_recognition_model_v1(
        'dlib_face_recognition_resnet_model_v1.dat')  # 人脸128D特征计算，不同人脸特征间可通过余弦相似度计算

    # f_128 = feature.compute_face_descriptor(image, detected_landmarks)
    p1 = detected_landmarks.part(1)
    p2 = detected_landmarks.part(15)
    p1 = tuple(eval(str(p1)))
    p2 = tuple(eval(str(p2)))
    image2 = img2mask(image, p1, p2)

    f_128 = feature.compute_face_descriptor(image2, detected_landmarks)
    # print('>' * 10, f_128)
    print('=' * 10, f_128)

    # for index, pt in enumerate(detected_landmarks.parts()):
    #     # print('Part {}: {}'.format(index, pt))
    #     pt_pos = (pt.x, pt.y)
    #     cv2.circle(image, pt_pos, 2, (255, 0, 0), 1)
    #
    # # 在新窗口中显示
    # cv2.imshow('aaa', image)
    # # print(detected_landmarks)
    # # for i, face in enumerate(res):
    # #     detected_landmarks = predictor(image, d).parts()
    # #     landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
