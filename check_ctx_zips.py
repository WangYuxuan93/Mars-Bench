#!/usr/bin/env python3
"""批量检查 CTX zip 完整性，统计损坏文件数量。

用法：
    python3 check_ctx_zips.py <dir1> [dir2 ...]

对每个目录下的 *.zip 做快速检查（读 zip 目录表 + 尝试打开内部 tif 的文件头，
每个文件秒级），最后输出统计结果，损坏文件清单写入当前目录 bad_zips.txt。
"""
import glob
import os
import sys
import zipfile


def check_zip(path: str):
    """返回 (是否完好, 错误信息)。检查的正是推理脚本里会失败的那一步。"""
    try:
        with zipfile.ZipFile(path) as zf:
            name = next((n for n in zf.namelist()
                         if n.lower().endswith((".tif", ".tiff"))), None)
            if name is None:
                return False, "zip 内没有 tif 文件"
            with zf.open(name) as fh:
                fh.read(16)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    all_zips = []
    for d in sys.argv[1:]:
        zips = sorted(glob.glob(os.path.join(d, "*.zip")))
        if not zips:
            print(f"警告: {d} 下没有找到 zip 文件")
        all_zips += zips

    ok_count = 0
    bad_list = []
    for i, p in enumerate(all_zips, 1):
        ok, err = check_zip(p)
        size_gb = os.path.getsize(p) / 1024 ** 3
        if ok:
            ok_count += 1
            print(f"[{i}/{len(all_zips)}] OK   {size_gb:6.2f}GB  {os.path.basename(p)}")
        else:
            bad_list.append((p, err))
            print(f"[{i}/{len(all_zips)}] BAD  {size_gb:6.2f}GB  {os.path.basename(p)}  ({err})")

    print()
    print(f"总计 {len(all_zips)} 个 zip：完好 {ok_count} 个，损坏 {len(bad_list)} 个")

    if bad_list:
        with open("bad_zips.txt", "w") as f:
            for p, _err in bad_list:
                f.write(p + "\n")
        print(f"损坏文件清单已写入: {os.path.abspath('bad_zips.txt')}")
        print("\n损坏文件：")
        for p, _err in bad_list:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
