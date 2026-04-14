# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [('assets', 'assets')]
datas += collect_data_files('imblearn')
datas += collect_data_files('xgboost')
datas += collect_data_files('lightgbm')
binaries = []
binaries += collect_dynamic_libs('xgboost')
binaries += collect_dynamic_libs('lightgbm')

a = Analysis(
    ['src\\mlquick_desktop.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MLquickDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets\\mlquick-logo.ico',
    version='version_info.txt',
)
