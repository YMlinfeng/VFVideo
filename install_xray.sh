#!/bin/bash
# bash install_xray.sh  # install on current host
# bash install_xray.sh all  # install on all hosts by mpirun


function print_green() { echo -e "\e[32m\e[100m$1\e[0m" >&2 ; }
function print_yellow() { echo -e "\e[33m\e[100m$1\e[0m" >&2 ; }
function print_red() { echo -e "\e[31m\e[100m$1\e[0m" >&2 ; }
# http代理从kml页面粘贴过来：https://kml.corp.kuaishou.com/v2/#/personal/workspace
export http_proxy=http://oversea-squid4.sgp.txyun:11080
export https_proxy=http://oversea-squid4.sgp.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
cloud_storage="https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects"
arg_install_nccl_dir=$1  #  ''：单机安装在/opt/kccl    '/full/path/xx'：单机安装在该路径    'all'：多机安装且优选ceph
arg_install_xray_pkg=$2  #  ''：单机自己下载安装        '/full/path/xx'：单机使用已下载好的文件安装
arg_install_xray_ver=$3  #  ''：单机自己查xray最新版本   '/full/path/xx'：单机使用已读取的xray版本（例如 0.3.28）
arg_install_kccl_ver=$4  #  ''：单机自己查kccl版本列表   '/full/path/xx'：单机使用已读取的kccl列表（例如 libkccl1 libkccl2）
arg_install_pip_cache=$5  #  ''：单机自己独立装          '/full/path'：单机使用传入的路径作为pip cache目录（复用内部下载好的文件，禁用网络）
arg_install_apt_cache=$6  #  ''：单机自己独立装          '/full/path'：单机使用传入的路径作为apt cache目录（复用内部下载好的文件，禁用网络）

if [ "$(cat /etc/xray_std_version 2>&1)" == "$(xray version 2>&1 | awk '{print $NF}')" ] && [ "$XRAY_INSTALL_FORCE" != "1" ]; then
    print_green "this docker image was already installed with standard xray version '$(cat /etc/xray_std_version)', skip updating"
    print_yellow "to forcely update xray (not fully tested in this env, mainly for debug), please 'export XRAY_INSTALL_FORCE=1' first"
    exit 0
fi
pip3_pkgs=(requests numpy pandas xlrd openpyxl plotly)
apt_pkgs1=(apt-utils net-tools iproute2 lldpd bind9-utils bind9-host ethtool iputils-ping sudo bc coreutils)
apt_pkgs2=(ibutils libtool libtool-bin librdmacm-dev libibumad-dev libpci-dev pciutils libudev-dev ibverbs-utils pkg-config)
yum_pkgs1=(autoconf automake libtool m4 net-tools iproute lldpd bind-utils ethtool iputils sudo bc coreutils)
yum_pkgs2=(ibutils libtool rdma-core-devel pciutils-devel libudev-devel libibverbs-utils pkgconfig)


# Step0: 检查OS版本和GPU版本
to_std_cuda_ver() { while IFS= read -r v; do        
    if [[ "$v" -ge 110 && "$v" -le 119 ]]; then echo "11"
    elif [[ "$v" -ge 120 && "$v" -le 127 ]]; then echo "12"
    elif [[ "$v" -ge 128 && "$v" -le 129 ]]; then echo "128"
    elif [[ "$v" -ge 130 && "$v" -le 139 ]]; then echo "13"
    else echo "$v"; fi; done
}
_os_=$(grep -Ei '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
if [ "${_os_}" == "ubuntu" ]; then _pkg_="deb"; elif [ "${_os_}" == "centos" ]; then _pkg_="rpm"
else print_red "Error: unsupported os for xray, only centos and ubuntu are available"; exit 1; fi
if command -v nvcc &> /dev/null; then
    _gpu_="cuda"$(nvcc --version | grep -oP 'release \K[\d\.]+' | tr -d '.' | to_std_cuda_ver)
elif nvcc=$(timeout 5s find /usr/ -type f -name nvcc -executable -print -quit) && [ -f "$nvcc" ]; then
    _gpu_="cuda"$("$nvcc" --version | grep -oP 'release \K[\d\.]+' | tr -d '.' | to_std_cuda_ver)
elif command -v nvidia-smi &> /dev/null; then
    _gpu_="cuda"$(nvidia-smi | grep -oP 'CUDA Version: \K[\d\.]+' | tr -d '.' | to_std_cuda_ver)
elif command -v amd-sli &> /dev/null || command -v rocm-sli &> /dev/null ; then
    _gpu_="rocm6"  # TODO: 动态识别
else print_red "Error: unrecognized cuda/rocm version"; exit 2
fi


function read_xray_latest() {
    if [ -z "$arg_install_xray_ver" ]; then  # 避免重复wget
        remote_url="${cloud_storage}/xray/VERSION"
        arg_install_xray_ver="$(wget -qO - $remote_url)"
        echo "reading $remote_url and get '$arg_install_xray_ver'" >&2
        if [ "$?" != '0' ] || [ -z "$arg_install_xray_ver" ]; then
            print_red "Error: failed to read latest xray from ${remote_url}"; exit 3;
        fi
    fi
    echo "$arg_install_xray_ver"
}


function download_xray_pkg() {
    set -e; local _name_; local _remote_
    if [ ! -f "${arg_install_xray_pkg}" ]; then  # 避免重复wget
        if [ -z "${XRAY_INSTALL_DEV}" ]; then
            _name_=xray_latest.${_gpu_}_amd64.${_pkg_}
            _remote_="${cloud_storage}/xray/${_name_}"
        elif [[ "${XRAY_INSTALL_DEV}" =~ ^http.+xray_.+$ ]]; then
            XRAY_INSTALL_DEV=$(sed 's/%2F/\//g' <<< "${XRAY_INSTALL_DEV}")
            _name_=$(awk -F'/' '{print $NF}' <<< "${XRAY_INSTALL_DEV}")
            _remote_="${XRAY_INSTALL_DEV}"
        else
            print_red "Error: XRAY_INSTALL_DEV should be like: https://...xray_....rpm"; exit 4;
        fi
        _download_="$(hostname -i)_${_name_}"
        echo "downloading xray pkg ${_download_} from ${_remote_} into dir $(pwd)" >&2 # outputs to stderr
        wget -qO "${_download_}" "${_remote_}" || { print_red "Error: download xray from ${_remote_} failed"; exit 4; }
        echo $(realpath "${_download_}")
    else
        echo "reusing existing xray pkg ${arg_install_xray_pkg} in dir $(pwd)" >&2 # outputs to stderr
        echo $(realpath "${arg_install_xray_pkg}")
    fi
}


function read_kccl_latests() {
    if [ -z "$arg_install_kccl_ver" ]; then  # 避免重复wget
        remote_url="${cloud_storage}/user-cloud-storage/nccl-kai/kccl/${_os_}-${_gpu_}/latests"
        arg_install_kccl_ver="$(wget -qO - $remote_url | tr '\n' ' ')"
        echo "reading $remote_url and get '$arg_install_kccl_ver'" >&2
        if [ "$?" != '0' ] || [ -z "$arg_install_kccl_ver" ]; then
            print_red "Error: failed to read latest kccl from ${remote_url}"; exit 5;
        fi
    fi
    echo "$arg_install_kccl_ver"
}


function apt_install_func() {
    if [ -f /etc/apt/sources.list ]; then # ubuntu 22
        cp /etc/apt/sources.list /etc/apt/sources.list.bak
        sed -i 's|http://|https://|g' /etc/apt/sources.list
    fi
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then # ubuntu 24
        cp /etc/apt/sources.list.d/ubuntu.sources /etc/apt/sources.list.d/ubuntu.sources.bak
        sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources
    fi
    mkdir -p /tmp; chmod 1777 /tmp;
    if [ "$apt_updated" != 1 ]; then
        if apt update &> /dev/null; then print_green "apt update success"; apt_updated=1
        elif apt update &> /dev/null; then print_green "apt update success"; apt_updated=1
        else print_red "apt update failed"; exit 6;
        fi
    fi
    dpkg --configure -a &> /dev/null
    # 主节点仅下载
    if [ "$apt_download_only" == 1 ]; then
        apt-get install -y --download-only --reinstall -o Dir::Cache::archives="$apt_cache_dir" "$@" &>/dev/null ||
            apt-get install -y --download-only --reinstall -o Dir::Cache::archives="$apt_cache_dir" "$@" &>/dev/null ||
                { print_red "ERROR: apt download $* failed"; exit 6; }
        print_green "apt pkgs $* downloaded into $apt_cache_dir"; return 0
    fi
    # 所有节点尝试用缓存安装
    if [ -d "$arg_install_apt_cache" ]; then
        all_cached=1; cached_debs=(); local found_deb
        for pkg in "$@"; do found_deb=$(ls "$arg_install_apt_cache"/${pkg}_*.deb 2>/dev/null | head -1);
            if [ -z "$found_deb" ]; then all_cached=false; break; fi; cached_debs+=("$found_deb")
        done
        if [ "$all_cached" = 1 ] && [ ${#cached_debs[@]} -gt 0 ]; then
            if apt-get install -y "${cached_debs[@]}" &> /dev/null; then
                print_green "apt install $* (from cache) success"; return 0
            elif apt-get -f install -y &> /dev/null && apt-get install -y "${cached_debs[@]}" &> /dev/null; then
                print_green "apt install $* (from cache) retry success"; return 0
            else print_yellow "apt install $* (from cache) failed"
            fi
        fi
    fi
    # 所有节点尝试用网络安装
    if apt-get install -y "$@" &> /dev/null; then
        print_green "apt install $* (from source) success"; return 0
    elif apt-get -f install -y &> /dev/null && apt-get install -y "$@" &> /dev/null; then
        print_green "apt install $* (from source) retry success"; return 0
    else print_yellow "apt install $* (from source) failed"
    fi
}
function yum_install_func() {
    if yum install -y --nogpgcheck --skip-broken $yum_pkgs &> /dev/null; then
        print_green "yum install $yum_pkgs (from source) success"; return 0
    elif yum install -y --nogpgcheck --skip-broken $yum_pkgs &> /dev/null; then
        print_green "yum install $yum_pkgs (from source) retry success"; return 0
    else print_red "yum install $yum_pkgs failed"
    fi
}


function setup_pip_venv() { # 创建python虚拟环境并安装依赖（主节点和所有从节点都会调用）
    SYSTEM_PYTHON3=$(python3 -c "import sys; print(getattr(sys, '_base_executable', sys.executable))" 2>/dev/null)
    [ -z "$SYSTEM_PYTHON3" ] && { print_red "ERROR: No system python3 found"; exit 7; }
    if ! command -v virtualenv &> /dev/null; then  # master需要预先安装虚拟环境，有利于下载pip包给worker使用
        local options="-q --break-system-packages --root-user-action=ignore"
        if [ ! -d "$arg_install_pip_cache" ]; then pip3 install $options virtualenv 2>/dev/null;
        else pip3 install $options --no-index --find-links="$arg_install_pip_cache" virtualenv 2>/dev/null; fi
    fi
    if [ ! -x /opt/venv/xray/bin/python3 ]; then # 使用非系统的pip和virtualenv新建其他venv是没问题的
        mkdir -p /opt/venv/ && command -v virtualenv &> /dev/null && virtualenv -p "$SYSTEM_PYTHON3" /opt/venv/xray | grep created
    fi
    if [ ! -x /opt/venv/xray/bin/python3 ] || [ ! -x /opt/venv/xray/bin/pip3 ]; then
        print_red "ERROR: Failed to create venv"; exit 7
    fi
}
setup_pip_venv # 提前准备虚拟环境
function install_pip_venv() {
    if [ ! -d "$arg_install_pip_cache" ]; then /opt/venv/xray/bin/pip3 install -q "${pip3_pkgs[@]}"
    else /opt/venv/xray/bin/pip3 install -q --no-index --find-links="$arg_install_pip_cache" "${pip3_pkgs[@]}"; fi
    grep -q plotly <<< "$(/opt/venv/xray/bin/pip3 list)" || print_yellow "WARN: pip install "${pip3_pkgs[*]}" failed"
}


function check_ssh() { # 这里避免高并发 wait pid; code=$? 因为会在某些容器中报错 pid not child
    local flag=ok batch=128 hostfile=/etc/mpi/mpi-hostfile _h_=() _l_ i n s e ip code
    if [ -f "$1" ]; then hostfile=$1; fi; if [[ "$2" =~ ^[0-9]+$ ]]; then batch=$2; fi;
    ssh_options='-o ConnectTimeout=2 -o PasswordAuthentication=no -o StrictHostKeyChecking=no -o LogLevel=ERROR'
    while read -r _l_; do _h_+=("$_l_"); done < <(grep -oE '^\s*([0-9]+\.){3}[0-9]+' "$hostfile")
    n=${#_h_[@]}; s=0; e=$((batch<n?batch:n)); while ((e>s)); do for ((i=s;i<e;i++)); do sleep 0.01
    { ip=${_h_[$i]}; ssh $ssh_options "$ip" true; echo $? > "ssh.$ip.code"; } & done 2>/dev/null; wait; s=$e; e=$((e+batch<n?e+batch:n)); done
    for ((i=0;i<n;i++)); do ip=${_h_[$i]}; code=$(cat "ssh.$ip.code" 2>&1); if [ "$code" != 0 ]; then echo "ssh $ip error: $code"; flag=err; fi; done
    rm -f ssh*code; if [ "$flag" == err ]; then return 1; else return 0; fi
}


# Step1: 多机安装，尝试选择可读写的最大可用量ceph网盘路径
if [ "$arg_install_nccl_dir" == "all" ]; then hostfile=/etc/mpi/mpi-hostfile
    for ((j=10; j>=0; j--)); do if check_ssh; then break; else sleep 2s; fi; if [ "$j" == 0 ]; then exit 7; fi; done
    ceph_df_info=$(xargs -I{} df -B 1G {} < <(awk '{print $3}' < <(grep -E "type ceph [^a-z]rw[^a-z]" < <(mount))))
    read -r ceph_dir ceph_size _ < <(tail -1 < <(sort -k2 -n < <(awk '!/^File/ {print $6,$4}' <<< "$ceph_df_info")))
    if [ -d "$ceph_dir" ] && [[ "$ceph_size" =~ ^[1-9][0-9]*$ ]]; then
        mkdir -p "${ceph_dir}/xray/" && cp "$0" "${ceph_dir}/xray/" && cd "${ceph_dir}/xray/"
        apt_cache_dir="${ceph_dir}/xray/apt_cache_${KML_ID}_${JOB_NAME}"; mkdir -p $apt_cache_dir; print_green "apt cache set to $apt_cache_dir"
        apt_download_only=1 apt_install_func "${apt_pkgs1[@]}" && apt_download_only=1 apt_install_func "${apt_pkgs2[@]}"
        pip_cache_dir="${ceph_dir}/xray/pip_cache_${KML_ID}_${JOB_NAME}"; mkdir -p $pip_cache_dir; print_green "pip cache set to $pip_cache_dir"
        if /opt/venv/xray/bin/pip3 download -q --dest "$pip_cache_dir" virtualenv "${pip3_pkgs[@]}"; then 
        print_green "pip pkgs downloaded into $pip_cache_dir"; else print_red "ERROR: pip3 download "${pip3_pkgs[*]}" failed"; exit 8; fi
    else
        unset ceph_dir; print_yellow "find no available ceph dir"
    fi
    TCP_NIC=$(grep -o "^\w*" < <(grep -B1 " ""$(hostname -i)"" " < <(ifconfig)))
    xray_ver="$(read_xray_latest)"; xray_pkg="$(download_xray_pkg)"; kccl_ver="$(read_kccl_latests)"
    if ! ifconfig "$TCP_NIC" &>/dev/null ; then print_red "ERROR: cannot find valid tcp nic"; exit 9; fi
    handle_output=' 2>&1 | sed "s/^/[$(hostname)] /g" || { echo -e "\e[31m\e[100m xray install failed on $(hostname) \e[0m" ; exit 1; }'
    mpirun_return="0"; PATH="$(sed -e 's/\/opt\/xray\/deps://' <<< "$PATH")"; mpirun=$(which mpirun)
    if [ -d "$ceph_dir" ]; then
        print_green "installing xray onto multi-hosts via ceph directory $ceph_dir/xray"
        env -i PATH="$PATH" HOME="$HOME" SUDO_USER="$SUDO_USER" OPAL_PREFIX="$OPAL_PREFIX" XRAY_INSTALL_DEV="$XRAY_INSTALL_DEV" XRAY_INSTALL_FORCE="$XRAY_INSTALL_FORCE" \
            timeout 300s $mpirun --allow-run-as-root -pernode -hostfile $hostfile -mca plm_rsh_args '-o LogLevel=ERROR' \
            -mca btl tcp,self -mca pml ob1 -mca btl_tcp_if_include $TCP_NIC -mca oob_tcp_if_include $TCP_NIC \
            -mca plm_rsh_num_concurrent 600 -mca routed_radix 600 -mca btl_openib_allow_ib false -mca orte_abort_on_non_zero_status 0 \
            -x PATH -x HOME -x SUDO_USER -x XRAY_INSTALL_DEV -x XRAY_INSTALL_FORCE \
            bash -c "set -e; set -o pipefail; bash ${ceph_dir}/xray/$0 '${ceph_dir}' '${xray_pkg}' '${xray_ver}' '${kccl_ver}' '${pip_cache_dir}' '${apt_cache_dir}' $handle_output"
    else # --preload-files --set-cwd-to-session-dir 这种用法可能在某些镜像会有未知原因报错 Epoll ADD(1) on fd xx failed 
        print_green "installing xray onto multi-hosts by mpirun preload-files"
        env -i PATH="$PATH" HOME="$HOME" SUDO_USER="$SUDO_USER" OPAL_PREFIX="$OPAL_PREFIX" XRAY_INSTALL_DEV="$XRAY_INSTALL_DEV" XRAY_INSTALL_FORCE="$XRAY_INSTALL_FORCE" \
            timeout 300s $mpirun --allow-run-as-root -pernode -hostfile $hostfile -mca plm_rsh_args '-o LogLevel=ERROR' \
            -mca btl tcp,self -mca pml ob1 -mca btl_tcp_if_include $TCP_NIC -mca oob_tcp_if_include $TCP_NIC \
            -mca plm_rsh_num_concurrent 600 -mca routed_radix 600 -mca btl_openib_allow_ib false -mca orte_abort_on_non_zero_status 0 \
            -x PATH -x HOME -x SUDO_USER -x XRAY_INSTALL_DEV -x XRAY_INSTALL_FORCE --preload-files "$0" --set-cwd-to-session-dir \
            bash -c "set -e; set -o pipefail; bash $0 '${ceph_dir}' '' '${xray_ver}' '${kccl_ver}' $handle_output"
    fi
    mpirun_return="$?"; rm -f "${xray_pkg}" &>/dev/null || true
    if [ "$mpirun_return" != "0" ]; then print_red "Error: mpirun xray install failed on some nodes"; exit 10; fi
    print_green "xray install success on multi-nodes, intermediate warnings can be ignored"; exit 0
fi


# Step2: 更新xray的依赖
if [ "${_os_}" == "ubuntu" ]; then echo "updating apt dependencies";
    apt_install_func "${apt_pkgs1[@]}" && apt_install_func "${apt_pkgs2[@]}" && lldpd && install_pip_venv
elif [ "${_os_}" == "centos" ]; then echo "updating yum dependencies";
    yum_install_func "${yum_pkgs1[@]}" && yum_install_func "${yum_pkgs2[@]}" && lldpd && install_pip_venv
fi


# Step3: 检查是否需要重装xray
up_to_date="0"
if [ -z "$arg_install_xray_ver" ]; then arg_install_xray_ver="$(read_xray_latest)"; fi
installed=$(xray version 2>/dev/null | awk '{print $NF}')
if [ "$arg_install_xray_ver" == "$installed" ] && [ -n "$installed" ] && [ -z "$XRAY_INSTALL_DEV" ] ; then
    print_green "xray is already up-to-date"; up_to_date="1"
else
    echo "trying to remove old version of xray"
    if [ "${_os_}" == "ubuntu" ]; then apt-get remove -y xray &> /dev/null || true
    else yum remove -y xray &> /dev/null || true; fi
fi


# Step4: 安装xray
if [ "$up_to_date" == "0" ]; then
    xray_pkg=$(download_xray_pkg)
    PATH=$(sed -e 's/:\/opt\/xray\/deps//g' -e 's/^\/opt\/xray\/deps://' <<< "$PATH")  # 排除上次安装卸载的影响
    which mpirun > /etc/xray_mpirun_path; which srun > /etc/xray_srun_path; which ray > /etc/xray_ray_path
    error="0"
    if [ "${_os_}" == "ubuntu" ]; then
        dpkg --configure -a &> /dev/null || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${xray_pkg}" >/dev/null \
        && print_green "xray apt install success" || { print_red "ERROR: xray apt install failed"; error=13; }
    else
        yum localinstall -y "${xray_pkg}" >/dev/null \
        && print_green "xray yum install success" || { print_red "ERROR: xray yum install failed"; error=13; }
    fi
    if [ ! -f "$arg_install_xray_pkg" ]; then rm -f "${xray_pkg}" &> /dev/null || true; fi
    if [ "$error" != "0" ]; then exit "$error"; fi
fi


# Step5: 现场编译gpu版本perftest，预编译版易出现不匹配的问题，编译时环境内多个版本也可能导致编译&链接版本不同因此需要pkg-config查询后指定
tar="perftest-4.5-0.20-cuda13.src.tar"
if [ -f "/opt/xray/$tar" ]; then
    pushd /opt/xray/ >/dev/null && tar -xvf "$tar" >/dev/null && cd perftest && ./autogen.sh &>/dev/null \
    && cflags=$(pkg-config --cflags libibverbs) && libs=$(pkg-config --libs libibverbs) \
    && ./configure CPPFLAGS="$cflags" LDFLAGS="$libs" CUDA_H_PATH=/usr/local/cuda/include/cuda.h &>/dev/null \
    && make -j &>/dev/null && cp ./ib_write_bw /opt/xray/deps/ \
    && print_green "perftest install success" || print_yellow "WARN: perftest install failed" # do not exit
fi


# Step6: 下载合适的NCCL库到共享目录或本地目录
mount_info=$(mount); rank="$OMPI_COMM_WORLD_RANK"
install_dir=$arg_install_nccl_dir
if [ ! -d "$install_dir" ]; then install_dir="/opt"; fi
if grep -q "$install_dir type ceph" <<< "$mount_info" && [[ $rank =~ ^[1-9][0-9]*$ ]]; then skip=1; fi
KCCL_PATH="$install_dir/kccl/${_os_}/${_gpu_}"
mkdir -p "$KCCL_PATH" && ls "$KCCL_PATH" >/dev/null && cd "$KCCL_PATH" || { print_red "ERROR: cannot enter $KCCL_PATH"; exit 14; }
kccls="$(read_kccl_latests)"  # TODO 打点版本集成版本
if [ "$skip" != "1" ]; then echo -e "downloading latest kccl into ${KCCL_PATH}"; fi
remote_url="${cloud_storage}/user-cloud-storage/nccl-kai/kccl/${_os_}-${_gpu_}/"
for kccl in $kccls; do
    if [ "$skip" != "1" ]; then
        if ! timeout 5s ldd $kccl &>/dev/null; then
            wget -qO "$kccl" "${remote_url}$kccl"
        fi
        if ! timeout 5s ldd $kccl >/dev/null; then
            print_red "ERROR: download from ${remote_url}$kccl to ${KCCL_PATH}/$kccl failed"; exit 15
        fi
    fi
    echo "${KCCL_PATH}/$kccl" > /etc/xray_kccl_path
done
if [ "$skip" != "1" ]; then echo "set $(cat /etc/xray_kccl_path) as default nccl"; fi
