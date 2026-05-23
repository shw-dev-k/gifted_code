import cv2

def draw_ui(image, status):

    scale = max(0.5, image.shape[1] / 1920.0)

    if status["help"]:

        if int(status["time"] * 2) % 2 == 0:

            cv2.putText(
                image,
                "!! HELP !!",
                (int(50 * scale), int(80 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.0 * scale,
                (0, 0, 255),
                max(1, int(7 * scale))
            )

    elif status["fall"]:

        cv2.putText(
            image,
            "FALL DETECTED",
            (int(50 * scale), int(80 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5 * scale,
            (0, 0, 255),
            max(1, int(5 * scale))
        )

        cv2.putText(
            image,
            f"Fall Time: {status['fall_duration']:.2f}s",
            (int(50 * scale), int(160 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0 * scale,
            (0, 165, 255),
            max(1, int(4 * scale))
        )

        cv2.putText(
            image,
            f"HELP until: {status['remaining']:.1f}s",
            (int(50 * scale), int(240 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8 * scale,
            (0, 255, 255),
            max(1, int(4 * scale))
        )

    cv2.putText(
        image,
        f"spine:{status['spine']:.1f} ratio:{status['ratio']:.2f} speed:{status['speed']:.1f}",
        (int(20 * scale), image.shape[0] - int(20 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2 * scale,
        (200, 200, 200),
        max(1, int(2 * scale))
    )
