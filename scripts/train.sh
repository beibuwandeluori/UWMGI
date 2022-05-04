DEVICE_ID=4
BATCH_SIZE=64
#python train.py --device_id=${DEVICE_ID} --batch_size=${BATCH_SIZE} --k=0
python train.py --device_id=${DEVICE_ID} --batch_size=${BATCH_SIZE} --k=1
python train.py --device_id=${DEVICE_ID} --batch_size=${BATCH_SIZE} --k=2
python train.py --device_id=${DEVICE_ID} --batch_size=${BATCH_SIZE} --k=3
python train.py --device_id=${DEVICE_ID} --batch_size=${BATCH_SIZE} --k=4