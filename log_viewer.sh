#!/bin/bash

# MLquick 日志查看脚本
# 方便查看和管理MLquick应用程序的日志

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_header() {
    echo -e "${CYAN}================================${NC}"
    echo -e "${CYAN}       MLquick 日志查看器${NC}"
    echo -e "${CYAN}================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "MLquick 日志查看脚本"
    echo ""
    echo "用法: $0 [选项] [参数]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -l, --list          列出所有日志文件"
    echo "  -t, --today         查看今天的日志"
    echo "  -y, --yesterday     查看昨天的日志"
    echo "  -d, --date DATE    查看指定日期的日志 (格式: YYYY-MM-DD)"
    echo "  -n, --tail NUM     显示最后N行"
    echo "  -f, --follow        实时跟踪日志文件"
    echo "  -s, --search TEXT   搜索日志内容"
    echo "  -e, --error         只显示错误日志"
    echo "  -w, --warning       只显示警告和错误日志"
    echo "  -c, --clear         清理旧日志文件"
    echo "  --stats              显示日志统计信息"
    echo ""
    echo "示例:"
    echo "  $0 -t                    # 查看今天的日志"
    echo "  $0 -n 100               # 显示最后100行"
    echo "  $0 -s '模型训练'       # 搜索包含'模型训练'的日志"
    echo "  $0 -e                    # 只显示错误日志"
    echo "  $0 -f                    # 实时跟踪日志"
    echo ""
}

# 列出所有日志文件
list_logs() {
    print_info "扫描日志目录: $LOG_DIR"

    if [ ! -d "$LOG_DIR" ]; then
        print_error "日志目录不存在: $LOG_DIR"
        exit 1
    fi

    echo "可用的日志文件:"
    echo "----------------------------------------"

    # 按日期排序显示日志文件
    find "$LOG_DIR" -name "*.log" -type f -printf "%f\n" | sort -r | while read -r logfile; do
        if [ -f "$logfile" ]; then
            local basename=$(basename "$logfile")
            local filesize=$(du -h "$logfile" | cut -f1)
            local modtime=$(stat -c %y "$logfile" 2>/dev/null || stat -f %Sm "$logfile" 2>/dev/null)
            local linecount=$(wc -l < "$logfile" 2>/dev/null || echo "未知")

            echo -e "📄 ${GREEN}$basename${NC}"
            echo "   大小: $filesize"
            echo "   修改时间: $modtime"
            echo "   行数: $linecount"
            echo "   路径: $logfile"
            echo "----------------------------------------"
        fi
    done

    if ! ls "$LOG_DIR"/*.log >/dev/null 2>&1; then
        print_warning "未找到任何日志文件"
    fi
}

# 显示指定日期的日志
show_date_log() {
    local date_str="$1"
    local logfile="$LOG_DIR/mlquick_${date_str}.log"

    if [ ! -f "$logfile" ]; then
        print_error "日志文件不存在: $logfile"
        return 1
    fi

    print_info "显示日志文件: $logfile"
    echo "----------------------------------------"

    # 使用less进行分页显示
    less -R +G "$logfile"
}

# 显示今天的日志
show_today_log() {
    local today=$(date +%Y%m%d)
    show_date_log "$today"
}

# 显示昨天的日志
show_yesterday_log() {
    local yesterday=$(date -d "yesterday" +%Y%m%d 2>/dev/null || date -v-1d +%Y%m%d)
    show_date_log "$yesterday"
}

# 显示最后N行
show_tail() {
    local lines="$1"
    local default_log="$LOG_DIR/mlquick_$(date +%Y%m%d).log"

    # 如果今天的日志不存在，找最新的日志文件
    if [ ! -f "$default_log" ]; then
        local latest_log=$(find "$LOG_DIR" -name "*.log" -type f -printf "%T@%p\n" | sort -n | tail -1 | cut -d@ -f2-)
        if [ -f "$latest_log" ]; then
            default_log="$latest_log"
        else
            print_error "未找到任何日志文件"
            return 1
        fi
    fi

    print_info "显示最后 $lines 行日志: $default_log"
    echo "----------------------------------------"

    # 带颜色显示最后N行
    tail -n "$lines" "$default_log" | while IFS= read -r line; do
        # 根据日志级别着色
        if [[ $line == *"ERROR"* ]]; then
            echo -e "${RED}$line${NC}"
        elif [[ $line == *"WARNING"* ]]; then
            echo -e "${YELLOW}$line${NC}"
        elif [[ $line == *"INFO"* ]]; then
            echo -e "${BLUE}$line${NC}"
        elif [[ $line == *"DEBUG"* ]]; then
            echo -e "${NC}$line${NC}"
        else
            echo "$line"
        fi
    done
}

# 实时跟踪日志
follow_log() {
    local default_log="$LOG_DIR/mlquick_$(date +%Y%m%d).log"

    # 如果今天的日志不存在，找最新的日志文件
    if [ ! -f "$default_log" ]; then
        local latest_log=$(find "$LOG_DIR" -name "*.log" -type f -printf "%T@%p\n" | sort -n | tail -1 | cut -d@ -f2-)
        if [ -f "$latest_log" ]; then
            default_log="$latest_log"
        else
            print_error "未找到任何日志文件"
            return 1
        fi
    fi

    print_info "实时跟踪日志: $default_log"
    print_info "按 Ctrl+C 退出跟踪"
    echo "----------------------------------------"

    # 使用tail -f跟踪，带颜色
    tail -f "$default_log" | while IFS= read -r line; do
        # 根据日志级别着色
        if [[ $line == *"ERROR"* ]]; then
            echo -e "${RED}$(date '+%H:%M:%S') - $line${NC}"
        elif [[ $line == *"WARNING"* ]]; then
            echo -e "${YELLOW}$(date '+%H:%M:%S') - $line${NC}"
        elif [[ $line == *"INFO"* ]]; then
            echo -e "${BLUE}$(date '+%H:%M:%S') - $line${NC}"
        elif [[ $line == *"DEBUG"* ]]; then
            echo -e "${NC}$(date '+%H:%M:%S') - $line${NC}"
        else
            echo "$(date '+%H:%M:%S') - $line"
        fi
    done
}

# 搜索日志内容
search_log() {
    local search_term="$1"
    local default_log="$LOG_DIR/mlquick_$(date +%Y%m%d).log"

    # 如果今天的日志不存在，搜索所有日志文件
    if [ ! -f "$default_log" ]; then
        print_info "在所有日志文件中搜索: $search_term"
        grep -n --color=always -i "$search_term" "$LOG_DIR"/*.log 2>/dev/null | while IFS=: read -r line_num content; do
            echo -e "${GREEN}$line_num${NC}: $content"
        done
    else
        print_info "在今天的日志中搜索: $search_term"
        grep -n --color=always -i "$search_term" "$default_log" 2>/dev/null | while IFS=: read -r line_num content; do
            echo -e "${GREEN}$line_num${NC}: $content"
        done
    fi
}

# 只显示错误日志
show_error_log() {
    local default_log="$LOG_DIR/mlquick_$(date +%Y%m%d).log"

    if [ ! -f "$default_log" ]; then
        local latest_log=$(find "$LOG_DIR" -name "*.log" -type f -printf "%T@%p\n" | sort -n | tail -1 | cut -d@ -f2-)
        if [ -f "$latest_log" ]; then
            default_log="$latest_log"
        else
            print_error "未找到任何日志文件"
            return 1
        fi
    fi

    print_info "显示错误日志: $default_log"
    echo "----------------------------------------"

    grep -n --color=always "ERROR" "$default_log" 2>/dev/null | while IFS=: read -r line_num content; do
        echo -e "${RED}$line_num${NC}: $content"
    done
}

# 只显示警告和错误日志
show_warning_error_log() {
    local default_log="$LOG_DIR/mlquick_$(date +%Y%m%d).log"

    if [ ! -f "$default_log" ]; then
        local latest_log=$(find "$LOG_DIR" -name "*.log" -type f -printf "%T@%p\n" | sort -n | tail -1 | cut -d@ -f2-)
        if [ -f "$latest_log" ]; then
            default_log="$latest_log"
        else
            print_error "未找到任何日志文件"
            return 1
        fi
    fi

    print_info "显示警告和错误日志: $default_log"
    echo "----------------------------------------"

    grep -n --color=always -E "(ERROR|WARNING)" "$default_log" 2>/dev/null | while IFS=: read -r line_num content; do
        if [[ $content == *"ERROR"* ]]; then
            echo -e "${RED}$line_num${NC}: $content"
        elif [[ $content == *"WARNING"* ]]; then
            echo -e "${YELLOW}$line_num${NC}: $content"
        fi
    done
}

# 清理旧日志文件
clean_logs() {
    print_info "扫描超过30天的日志文件..."

    local deleted_count=0
    local deleted_size=0

    # 删除超过30天的日志文件
    find "$LOG_DIR" -name "*.log" -type f -mtime +30 -print0 | while IFS= read -r -d $'\0' logfile; do
        if [ -f "$logfile" ]; then
            local filesize=$(du -k "$logfile" | cut -f1)
            print_warning "删除旧日志文件: $(basename "$logfile") (${filesize}KB)"
            rm "$logfile"
            deleted_count=$((deleted_count + 1))
            deleted_size=$((deleted_size + filesize))
        fi
    done

    if [ $deleted_count -gt 0 ]; then
        print_info "清理完成: 删除了 $deleted_count 个文件，释放了 ${deleted_size}KB 空间"
    else
        print_info "没有找到需要清理的旧日志文件"
    fi
}

# 显示日志统计信息
show_stats() {
    print_info "分析日志目录: $LOG_DIR"
    echo "========================================"

    if [ ! -d "$LOG_DIR" ]; then
        print_error "日志目录不存在"
        return 1
    fi

    # 统计信息
    local total_files=$(find "$LOG_DIR" -name "*.log" -type f | wc -l)
    local total_size=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
    local total_lines=0
    local total_errors=0
    local total_warnings=0
    local total_operations=0

    # 分析所有日志文件
    find "$LOG_DIR" -name "*.log" -type f -print0 | while IFS= read -r -d $'\0' logfile; do
        if [ -f "$logfile" ]; then
            local lines=$(wc -l < "$logfile" 2>/dev/null || echo 0)
            local errors=$(grep -c "ERROR" "$logfile" 2>/dev/null || echo 0)
            local warnings=$(grep -c "WARNING" "$logfile" 2>/dev/null || echo 0)
            local operations=$(grep -c "OPERATION" "$logfile" 2>/dev/null || echo 0)

            total_lines=$((total_lines + lines))
            total_errors=$((total_errors + errors))
            total_warnings=$((total_warnings + warnings))
            total_operations=$((total_operations + operations))

            echo -e "📄 ${GREEN}$(basename "$logfile")${NC}"
            echo "   行数: $lines"
            echo "   错误: $errors"
            echo "   警告: $warnings"
            echo "   操作: $operations"
            echo ""
        fi
    done

    echo "========================================"
    echo -e "总计统计:"
    echo -e "📁 日志文件数: ${GREEN}$total_files${NC}"
    echo -e "💾 总大小: ${GREEN}$total_size${NC}"
    echo -e "📝 总行数: ${GREEN}$total_lines${NC}"
    echo -e "❌ 总错误数: ${RED}$total_errors${NC}"
    echo -e "⚠️  总警告数: ${YELLOW}$total_warnings${NC}"
    echo -e "⚙️  总操作数: ${GREEN}$total_operations${NC}"
    echo "========================================"
}

# 主逻辑
main() {
    # 确保日志目录存在
    mkdir -p "$LOG_DIR"

    case "${1:-help}" in
        -h|--help)
            show_help
            ;;
        -l|--list)
            list_logs
            ;;
        -t|--today)
            show_today_log
            ;;
        -y|--yesterday)
            show_yesterday_log
            ;;
        -d|--date)
            if [ -z "$2" ]; then
                print_error "请提供日期参数 (格式: YYYY-MM-DD)"
                exit 1
            fi
            show_date_log "$2"
            ;;
        -n|--tail)
            if [ -z "$2" ]; then
                show_tail 50  # 默认显示50行
            else
                show_tail "$2"
            fi
            ;;
        -f|--follow)
            follow_log
            ;;
        -s|--search)
            if [ -z "$2" ]; then
                print_error "请提供搜索内容"
                exit 1
            fi
            search_log "$2"
            ;;
        -e|--error)
            show_error_log
            ;;
        -w|--warning)
            show_warning_error_log
            ;;
        -c|--clear)
            clean_logs
            ;;
        --stats)
            show_stats
            ;;
        *)
            print_header
            print_error "未知选项: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"