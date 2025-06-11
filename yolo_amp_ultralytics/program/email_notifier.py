import socket
import smtplib
from email.mime.text import MIMEText
import time
import swanlab
from swanlab.plugin.notification import EmailCallback

# --------------------------
# 配置区（使用前必须修改！）
# --------------------------
SMTP_CFG = {
    "sender": "727325858@qq.com",       # 发件邮箱
    "receiver": "727325858@qq.com",     # 收件邮箱
    "password": "ndweknfmqhefbbga",     # QQ邮箱授权码
    "server": "smtp.qq.com"             # SMTP服务器
}

def check_net():
    """检查网络连通性"""
    try:
        socket.create_connection(("www.baidu.com", 80), timeout=5)
        return True
    except Exception as e:
        print(f"❌ 网络不通：{str(e)}")
        return False

def send_mail(subject, content, retries=3):
    """优化版邮件发送函数（带重试和端口自动探测）"""
    if not check_net():
        return False

    for port in [465, 587]:  # 优先尝试465（SSL），再试587（STARTTLS）
        try:
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_CFG["server"], port)
            else:
                server = smtplib.SMTP(SMTP_CFG["server"], port)
                server.starttls()

            # 登录邮箱
            server.login(SMTP_CFG["sender"], SMTP_CFG["password"])

            # 构造邮件内容
            msg = MIMEText(content, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = SMTP_CFG["sender"]
            msg["To"] = SMTP_CFG["receiver"]

            # 发送邮件
            server.send_message(msg)
            print(f"✅ 邮件通过端口 {port} 发送成功！")
            return True
        except smtplib.SMTPException as smtp_err:
            print(f"⚠ SMTP异常：{smtp_err}")
        except Exception as e:
            print(f"⚠ 其他异常：{e}")
        finally:
            try:
                server.quit()
            except Exception as quit_err:
                print(f"⚠ 连接关闭异常：{quit_err}")

    # 全局重试逻辑（等待60秒后重试）
    for i in range(retries):
        print(f"⏳ 第 {i+1} 次重试（等待60秒）...")
        time.sleep(60)
        if send_mail(subject, content, retries=0):
            return True
    print("❌ 邮件发送最终失败！")
    return False

class RobustEmailCallback(EmailCallback):
    """增强版邮件回调插件（集成重试和网络检查）"""
    def __init__(self):
        # 调用父类构造函数，传递必要参数
        super().__init__(
            sender_email=SMTP_CFG["sender"],
            receiver_email=SMTP_CFG["receiver"],
            password=SMTP_CFG["password"],
            smtp_server=SMTP_CFG["server"],
            port=587,  # 默认使用587端口
            language="zh"
        )
    
    def _send_email(self, subject, content):
        """封装发送逻辑，使用优化的send_mail函数"""
        try:
            if send_mail(subject, content):
                print("📧 邮件通知准备完毕，已成功发送！")
                return True
            else:
                print("❌ 邮件通知准备失败，发送未成功。")
                return False
        except Exception as e:
            print(f"❌ 邮件发送异常：{e}")
            return False
    
    def on_train_end(self, trainer):
        """训练结束后发送邮件"""
        subject = f"训练完成：{trainer.project}/{trainer.name}"
        content = f"实验 {trainer.name} 已完成，查看链接：{trainer.run_url}"
        self._send_email(subject, content)
    
    def on_train_error(self, trainer, error):
        """训练出错时发送邮件"""
        subject = f"训练失败：{trainer.project}/{trainer.name}"
        content = f"实验 {trainer.name} 出现错误：{str(error)}"
        self._send_email(subject, content)

def initialize_swanlab(project_name, experiment_name=None):
    """
    初始化 SwanLab 实验并集成邮件通知插件。
    
    参数：
        project_name (str): SwanLab 项目的名称。
        experiment_name (str, optional): 实验的名称。如果未提供，则使用默认值。
    
    返回：
        bool: 初始化是否成功。
    """
    try:
        # 创建增强版邮件回调
        email_callback = RobustEmailCallback()
        
        # 初始化 SwanLab 实验
        swanlab.init(
            project=project_name,
            experiment=experiment_name or "default_exp",
            callbacks=[email_callback]
        )
        
        print(f"✅ 初始化成功！项目：{project_name}，实验：{experiment_name}")
        return True
    except Exception as e:
        print(f"❌ 初始化失败：{str(e)}")
        return False
    
# -----------------------------------
# 测试入口（直接运行此脚本时触发）
if __name__ == "__main__":
    # 测试邮件发送功能（无需依赖模型训练）
    test_subject = "测试邮件"
    test_content = "这是一封测试邮件，用于验证发送逻辑。"
    
    print("▶️ 开始测试邮件发送功能...")
    if send_mail(test_subject, test_content):
        print("✅ 邮件发送测试成功！")
    else:
        print("❌ 邮件发送测试失败，请检查配置！")