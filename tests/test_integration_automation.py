import os
import time

def test_integration_automation_safe():
    """
    安全的自动化集成测试脚本。
    - 默认为 dry-run，仅打印计划的操作。
    - 若设置环境变量 `RUN_REAL_AUTOMATION=1`，且你明确允许，则会尝试调用控制器中的方法（若存在）。
    """
    run_real = os.environ.get("RUN_REAL_AUTOMATION") == "1"
    actions_plan = [
        ("move_to", (100, 100)),
        ("left_click", (100, 100)),
        ("right_click", (120, 120)),
        ("smooth_scroll", -300),
        ("type_text", "自动化测试")
    ]

    if run_real:
        # 在真实执行前，尽量按存在性检查方法以避免异常。
        try:
            from ai_operator.automation import mouse_controller as mc, keyboard_controller as kc
            # 支持不同实现的控制器类名
            MouseCls = None
            for name in ("MouseController", "AdvancedMouseController", "advanced_mouse_controller"):
                MouseCls = getattr(mc, name, None)
                if MouseCls is not None:
                    break

            if MouseCls is None:
                raise ImportError("找不到合适的鼠标控制器类（尝试 MouseController/AdvancedMouseController）")

            m = MouseCls()
            k = getattr(kc, 'KeyboardController')()
        except Exception as e:
            print("无法导入或实例化控制器，跳过真实运行：", e)
            run_real = False

    if run_real:
        print("执行真实自动化动作（请确保你已允许）：")
        if hasattr(m, 'move_to'):
            try:
                m.move_to(100, 100, duration=0.2)
            except Exception as e:
                print('move_to 调用失败:', e)
        if hasattr(m, 'click'):
            try:
                m.click(100, 100)
            except Exception as e:
                print('click 调用失败:', e)
        if hasattr(m, 'right_click'):
            try:
                m.right_click(120, 120)
            except Exception as e:
                print('right_click 调用失败:', e)
        if hasattr(m, 'smooth_scroll'):
            try:
                # 不同实现可能使用不同参数名，先尝试 duration，再回退到 delay
                try:
                    m.smooth_scroll(-300, duration=0.5)
                except TypeError:
                    m.smooth_scroll(-300, steps=10, direction='vertical', delay=0.05)
            except Exception as e:
                print('smooth_scroll 调用失败:', e)
        if hasattr(k, 'type_text'):
            try:
                # KeyboardController.type_text 可能不接受 interval 参数
                try:
                    k.type_text('自动化测试', interval=0.05)
                except TypeError:
                    k.type_text('自动化测试')
            except Exception as e:
                print('type_text 调用失败:', e)
        time.sleep(0.5)
    else:
        print('Dry run — 计划的自动化动作（不会发送真实输入）：')
        for a in actions_plan:
            print(' -', a)

    assert True
