###
 # @Author: zhouyuchong
 # @Date: 2024-05-24 14:58:06
 # @Description: 
 # @LastEditors: zhouyuchong
 # @LastEditTime: 2024-05-30 16:51:08
### 
log() {
    local message="$1"
    current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[${current_time}] $message"
}

CUR_DIR=$PWD

log "[Start]Installing dependencies..."

log "[Check]Checking path..."
if [ ! -f "$CUR_DIR/readme.md" ]; then
    log "[ERROR]wrong root path!"
    exit 1
fi

log "[Info]build docker image..."
cd $CUR_DIR
ls
docker build -t kbrain:demo .

# pip3 install -i https://pypi.mirrors.ustc.edu.cn/simple/ -r requirments.txt

# python3 main.py /dev/video0