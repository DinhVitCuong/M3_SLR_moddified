python realtime_infer.py --config configs/Uniformer/test_cfg/UFOneView_MultiVSL200.yaml --checkpoint checkpoint/uniformer_VSL.pth --label-map data\MultiVSL200\lookuptable.csv --source 0 

python realtime_infer.py --config configs/Uniformer/realtime_VSL.yaml --checkpoint checkpoint/uniformer_VSL.pth --label-map data\MultiVSL200\lookuptable.csv --source 0 