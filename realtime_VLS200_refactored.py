    
'''
Nhớ sửa biến default của label-map-csv
Bật crop thì nhớ thêm tag "--use-slr-crop" vào mỗi cái script
Có các input mode:
    "--input-mode webcam" thì nhớ bỏ thêm biến "--camera-id "
    "--input-mode video" thì nhớ bỏ thêm biến "--video-path"
    Còn riêng với dùng esp32 cam thì
    "--input-mode webcam" + "--camera-url"
'''
# realtime webcam, cameraid:
#python realtime_VLS200_cropped.py --input-mode webcam --camera-id 1 --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --use-slr-crop --show

# realtime webcam, camera_url
#python realtime_VLS200_cropped.py --input-mode webcam --camera-url "http://192.168.1.123:81/stream" --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --use-slr-crop --show

# video
#python realtime_VLS200_cropped.py --input-mode video --video-path "Z:\SignLanguageReg\VSL_data\New folder\14_Bao-Nam_1-200_13-14-15_0112___center_device19_signer14_center_ord1_36.mp4" --checkpoint "Z:/SignLanguageReg/M3-SLR/checkpoint/uniformer_VSL.pth" --show 


from __future__ import annotations

from vsl_realtime_refactor import RealtimeVSLApp, build_parser, prepare_runtime


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runtime = prepare_runtime(args)
    app = RealtimeVSLApp(runtime)
    app.run()


if __name__ == "__main__":
    main()
