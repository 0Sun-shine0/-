"""
Web控制器 - 处理浏览器自动化操作
"""
import time
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from typing import Dict, List, Optional


class WebController:
    """Web浏览器自动化控制器"""
    
    def __init__(self):
        self.driver = None
        self.wait = None
        
    def start_browser(self, headless: bool = False):
        """启动浏览器"""
        try:
            options = Options()
            if headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            self.driver = webdriver.Chrome(options=options)
            self.wait = WebDriverWait(self.driver, 10)
            return True
        except Exception as e:
            print(f"启动浏览器失败: {e}")
            return False
    
    def open_url(self, url: str):
        """打开网页"""
        if self.driver:
            self.driver.get(url)
            time.sleep(2)  # 等待页面加载
            return True
        return False
    
    def find_element(self, selector: str, by: str = "css"):
        """查找网页元素"""
        if not self.driver:
            return None
            
        try:
            if by == "css":
                return self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            elif by == "id":
                return self.wait.until(EC.presence_of_element_located((By.ID, selector)))
            elif by == "name":
                return self.wait.until(EC.presence_of_element_located((By.NAME, selector)))
            elif by == "xpath":
                return self.wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            elif by == "class":
                return self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, selector)))
        except:
            return None
    
    def click_element(self, selector: str, by: str = "css"):
        """点击网页元素"""
        element = self.find_element(selector, by)
        if element:
            element.click()
            return True
        return False
    
    def input_text(self, selector: str, text: str, by: str = "css"):
        """在网页元素中输入文本"""
        element = self.find_element(selector, by)
        if element:
            element.clear()
            element.send_keys(text)
            return True
        return False
    
    def get_page_title(self) -> str:
        """获取页面标题"""
        if self.driver:
            return self.driver.title
        return ""
    
    def get_element_text(self, selector: str, by: str = "css") -> str:
        """获取元素文本"""
        element = self.find_element(selector, by)
        if element:
            return element.text
        return ""
    
    def close_browser(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            self.driver = None


class MultiAppController:
    """多应用控制器"""
    
    def __init__(self):
        self.active_apps = {}
        
    def start_excel(self):
        """启动Excel"""
        import subprocess
        try:
            subprocess.Popen(['start', 'excel'], shell=True)
            time.sleep(3)  # 等待Excel启动
            return True
        except:
            # 尝试其他Excel路径
            excel_paths = [
                r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\EXCEL.EXE",
                r"C:\Program Files\Microsoft Office\Office16\EXCEL.EXE",
                r"C:\Program Files (x86)\Microsoft Office\Office16\EXCEL.EXE",
            ]
            
            for path in excel_paths:
                try:
                    subprocess.Popen([path])
                    time.sleep(3)
                    return True
                except:
                    continue
            return False
    
    def start_powerpoint(self):
        """启动PowerPoint"""
        import subprocess
        try:
            subprocess.Popen(['start', 'powerpnt'], shell=True)
            time.sleep(3)  # 等待PowerPoint启动
            return True
        except:
            ppt_paths = [
                r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE",
                r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE",
                r"C:\Program Files\Microsoft Office\Office16\POWERPNT.EXE",
                r"C:\Program Files (x86)\Microsoft Office\Office16\POWERPNT.EXE",
            ]
            
            for path in ppt_paths:
                try:
                    subprocess.Popen([path])
                    time.sleep(3)
                    return True
                except:
                    continue
            return False
    
    def start_notepad(self):
        """启动记事本"""
        import subprocess
        try:
            subprocess.Popen(['notepad'])
            time.sleep(1)
            return True
        except:
            return False
    
    def start_application(self, app_name: str):
        """启动指定应用"""
        app_name = app_name.lower()
        if 'excel' in app_name or 'xls' in app_name:
            return self.start_excel()
        elif 'powerpoint' in app_name or 'ppt' in app_name:
            return self.start_powerpoint()
        elif 'notepad' in app_name or 'txt' in app_name:
            return self.start_notepad()
        elif 'word' in app_name or 'doc' in app_name:
            # 使用已有的Word启动逻辑
            import subprocess
            try:
                subprocess.Popen(['start', 'winword'], shell=True)
                time.sleep(3)
                return True
            except:
                return False
        return False


class DataProcessor:
    """数据处理自动化"""
    
    def __init__(self):
        pass
    
    def extract_data_from_text(self, text: str) -> Dict:
        """从文本中提取数据"""
        import re
        
        # 提取数字
        numbers = re.findall(r'\d+\.?\d*', text)
        # 提取邮箱
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        # 提取网址
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        # 提取日期
        dates = re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2}', text)
        
        return {
            'numbers': [float(n) for n in numbers if n],
            'emails': emails,
            'urls': urls,
            'dates': dates,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def process_data(self, data: Dict) -> Dict:
        """处理数据"""
        processed = {}
        
        if 'numbers' in data and data['numbers']:
            numbers = data['numbers']
            processed['sum'] = sum(numbers)
            processed['average'] = sum(numbers) / len(numbers)
            processed['max'] = max(numbers)
            processed['min'] = min(numbers)
        
        return processed
    
    def generate_report(self, data: Dict, title: str = "数据报告") -> str:
        """生成数据报告"""
        report = f"=== {title} ===\n\n"
        
        if 'numbers' in data:
            report += f"数字统计:\n"
            report += f"  总和: {data.get('sum', 0)}\n"
            report += f"  平均值: {data.get('average', 0)}\n"
            report += f"  最大值: {data.get('max', 0)}\n"
            report += f"  最小值: {data.get('min', 0)}\n\n"
        
        if 'emails' in data:
            report += f"邮箱地址: {len(data['emails'])} 个\n"
            for email in data['emails']:
                report += f"  - {email}\n"
            report += "\n"
        
        if 'urls' in data:
            report += f"网址链接: {len(data['urls'])} 个\n"
            for url in data['urls']:
                report += f"  - {url}\n"
            report += "\n"
        
        if 'dates' in data:
            report += f"日期: {len(data['dates'])} 个\n"
            for date in data['dates']:
                report += f"  - {date}\n"
            report += "\n"
        
        report += f"总计: {data.get('word_count', len(data.get('numbers', [])) + len(data.get('emails', [])) + len(data.get('urls', [])) + len(data.get('dates', [])))} 词, {data.get('char_count', len(str(data)))} 字符\n"
        
        return report


# 测试函数
def test_web_controller():
    """测试Web控制器"""
    print("测试Web控制器...")
    controller = WebController()
    
    if controller.start_browser():
        print("浏览器启动成功")
        controller.open_url("https://www.baidu.com")
        print(f"页面标题: {controller.get_page_title()}")
        controller.close_browser()
        print("浏览器测试完成")
    else:
        print("浏览器启动失败")


def test_multi_app_controller():
    """测试多应用控制器"""
    print("测试多应用控制器...")
    controller = MultiAppController()
    
    # 测试启动记事本
    if controller.start_notepad():
        print("记事本启动成功")
    else:
        print("记事本启动失败")


def test_data_processor():
    """测试数据处理器"""
    print("测试数据处理器...")
    processor = DataProcessor()
    
    sample_text = """
    联系方式：contact@example.com
    网站：https://www.example.com
    日期：2024-01-01
    价格：$99.99
    数量：100
    电话：400-123-4567
    """
    
    extracted = processor.extract_data_from_text(sample_text)
    print(f"提取的数据: {extracted}")
    
    processed = processor.process_data(extracted)
    print(f"处理后的数据: {processed}")
    
    report = processor.generate_report(processed, "测试报告")
    print(f"生成的报告:\n{report}")


if __name__ == "__main__":
    test_web_controller()
    test_multi_app_controller()
    test_data_processor()