#!/bin/bash

# MLquick 启动脚本
# 支持快速启动、停止和状态查看

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="MLquick"
MAIN_SCRIPT="src/MLquick.py"
PID_FILE="$SCRIPT_DIR/.mlquick.pid"
LOG_FILE="$SCRIPT_DIR/.mlquick.log"
DEFAULT_PORT=20040

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python和Streamlit是否安装
check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装，请先安装Python3"
        exit 1
    fi

    # 检查虚拟环境中的streamlit
    local streamlit_cmd=""
    if [ -d "$SCRIPT_DIR/venv" ]; then
        streamlit_cmd="$SCRIPT_DIR/venv/bin/python -m streamlit"
    elif [ -d "$SCRIPT_DIR/.venv" ]; then
        streamlit_cmd="$SCRIPT_DIR/.venv/bin/python -m streamlit"
    else
        streamlit_cmd="python3 -m streamlit"
    fi

    if ! $streamlit_cmd --help &> /dev/null; then
        print_error "Streamlit 未安装，请运行: pip install streamlit"
        exit 1
    fi
}

# 检查是否存在Python虚拟环境
check_venv() {
    if [ -d "$SCRIPT_DIR/venv" ]; then
        source "$SCRIPT_DIR/venv/bin/activate"
        print_info "虚拟环境已激活"
    elif [ -d "$SCRIPT_DIR/.venv" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
        print_info "虚拟环境已激活"
    fi
}

# 获取进程状态
get_status() {
    local pid=""

    # 首先尝试从PID文件获取
    if [ -f "$PID_FILE" ]; then
        local saved_pid=$(cat "$PID_FILE")
        if ps -p "$saved_pid" > /dev/null 2>&1; then
            echo "running"
            echo "$saved_pid"
            return 0
        else
            rm -f "$PID_FILE"
        fi
    fi

    # 如果PID文件不存在或无效，尝试通过端口查找进程
    local port_process=$(netstat -tlnp 2>/dev/null | grep ":$DEFAULT_PORT" | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$port_process" ] && ps -p "$port_process" > /dev/null 2>&1; then
        # 检查是否是streamlit进程
        local cmd=$(ps -o cmd= -p "$port_process" 2>/dev/null | grep -i streamlit)
        if [ -n "$cmd" ]; then
            echo "$port_process" > "$PID_FILE"
            echo "running"
            echo "$port_process"
            return 0
        fi
    fi

    echo "stopped"
    echo ""
    return 1
}

# 启动服务
start_service() {
    local port=${1:-$DEFAULT_PORT}

    print_info "检查依赖..."
    check_dependencies

    print_info "检查虚拟环境..."
    check_venv

    local status_output=($(get_status))
    local status=${status_output[0]}
    local pid=${status_output[1]}

    if [ "$status" = "running" ] && [ -n "$pid" ]; then
        print_warning "$PROJECT_NAME 已经在运行 (PID: $pid)"
        print_info "访问地址: http://localhost:$port"
        return 0
    fi

    print_info "启动 $PROJECT_NAME 服务..."
    print_info "端口: $port"

    # 切换到项目目录
    cd "$SCRIPT_DIR"

    # 检查主脚本是否存在
    if [ ! -f "$MAIN_SCRIPT" ]; then
        print_error "主脚本 $MAIN_SCRIPT 不存在"
        exit 1
    fi

    # 启动Streamlit应用
    nohup streamlit run "$MAIN_SCRIPT" \
        --server.port=$port \
        --server.headless=true \
        --server.address=0.0.0.0 \
        --browser.gatherUsageStats=false \
        --logger.level=info \
        > "$LOG_FILE" 2>&1 &

    local pid=$!
    echo $pid > "$PID_FILE"

    # 等待启动
    sleep 3

    # 验证启动是否成功
    status_output=($(get_status))
    status=${status_output[0]}

    if [ "$status" = "running" ]; then
        print_success "$PROJECT_NAME 启动成功!"
        print_success "PID: $pid"
        print_success "访问地址: http://localhost:$port"
        print_info "日志文件: $LOG_FILE"
    else
        print_error "$PROJECT_NAME 启动失败"
        print_error "请检查日志文件: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# 停止服务
stop_service() {
    local status_output=($(get_status))
    local status=${status_output[0]}
    local pid=${status_output[1]}

    if [ "$status" != "running" ] || [ -z "$pid" ]; then
        print_warning "$PROJECT_NAME 未在运行"
        return 0
    fi

    print_info "停止 $PROJECT_NAME (PID: $pid)..."

    # 尝试优雅停止
    kill "$pid" 2>/dev/null

    # 等待进程结束
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    # 如果进程仍在运行，强制终止
    if ps -p "$pid" > /dev/null 2>&1; then
        print_warning "优雅停止失败，强制终止进程..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    # 清理PID文件
    rm -f "$PID_FILE"

    # 验证停止状态
    status_output=($(get_status))
    status=${status_output[0]}

    if [ "$status" = "stopped" ]; then
        print_success "$PROJECT_NAME 已停止"
    else
        print_error "$PROJECT_NAME 停止失败"
        exit 1
    fi
}

# 查看状态
show_status() {
    local status_output=($(get_status))
    local status=${status_output[0]}
    local pid=${status_output[1]}

    echo "=========================================="
    echo "      $PROJECT_NAME 服务状态"
    echo "=========================================="

    if [ "$status" = "running" ] && [ -n "$pid" ]; then
        echo -e "状态: ${GREEN}运行中${NC}"
        echo "PID: $pid"

        # 获取端口信息
        local port=$(lsof -i -P -n | grep LISTEN | grep "$pid" | awk '{print $9}' | cut -d':' -f2)
        if [ -n "$port" ]; then
            echo -e "访问地址: ${BLUE}http://localhost:$port${NC}"
        else
            echo -e "访问地址: ${BLUE}http://localhost:$DEFAULT_PORT${NC} (默认)"
        fi

        # 显示运行时间
        local start_time=$(ps -o lstart= -p "$pid" | xargs)
        echo "启动时间: $start_time"

        # 显示内存使用
        local memory=$(ps -o rss= -p "$pid" | awk '{print int($1/1024)"MB"}')
        echo "内存使用: $memory"

    else
        echo -e "状态: ${RED}未运行${NC}"
    fi

    echo "=========================================="

    # 显示日志文件信息
    if [ -f "$LOG_FILE" ]; then
        local log_size=$(du -h "$LOG_FILE" | cut -f1)
        echo "日志文件: $LOG_FILE (大小: $log_size)"
    else
        echo "日志文件: 不存在"
    fi

    echo "=========================================="
}

# 查看日志
show_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        print_error "日志文件不存在: $LOG_FILE"
        exit 1
    fi

    print_info "显示最近50行日志:"
    echo "=========================================="
    tail -50 "$LOG_FILE"
    echo "=========================================="
}

# 重启服务
restart_service() {
    local port=${1:-$DEFAULT_PORT}
    print_info "重启 $PROJECT_NAME 服务..."
    stop_service
    sleep 2
    start_service "$port"
}

# 清理临时文件
cleanup() {
    print_info "清理临时文件..."
    rm -f "$PID_FILE"
    rm -f "$LOG_FILE"
    print_success "临时文件已清理"
}

# 显示帮助信息
show_help() {
    echo "MLquick 管理脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start [port]     启动服务 (默认端口: $DEFAULT_PORT)"
    echo "  stop            停止服务"
    echo "  restart [port]  重启服务"
    echo "  status          查看服务状态"
    echo "  logs            查看日志"
    echo "  cleanup         清理临时文件"
    echo "  help            显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start 8080          # 在8080端口启动"
    echo "  $0 restart             # 重启服务"
    echo "  $0 status              # 查看状态"
    echo "  $0 logs               # 查看日志"
    echo ""
}

# 主逻辑
main() {
    case "${1:-help}" in
        start)
            start_service "$2"
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service "$2"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 捕获中断信号
trap cleanup EXIT

# 执行主函数
main "$@"