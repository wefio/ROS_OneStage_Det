# email_notifier.py
import smtplib, socket, time
from email.mime.text import MIMEText

#============= 配置区 =============（修改这里！）
SMTP_CFG = {
    "sender": "example@qq.com",       # 发件邮箱
    "receiver": "example@qq.com",     # 收件邮箱
    "password": "example",     # QQ邮箱授权码
    "server": "smtp.qq.com"             # SMTP服务器
}

def check_net():
    """更简单的网络检查"""
    try:
        socket.create_connection(("www.baidu.com", 80), timeout=5)
        return True
    except:
        print("❌ 网络不通，请检查网络连接！")
        return False

def send_mail(subject, content, retries=3):
    """优化版发送逻辑"""
    if not check_net(): return False
    
    # 端口自动探测逻辑
    for port in [465, 587]:         # 先试465，失败改587
        try:
            # 动态选择加密方式
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_CFG["server"], port)
            else:
                server = smtplib.SMTP(SMTP_CFG["server"], port)
                server.starttls()
            
            server.login(SMTP_CFG["sender"], SMTP_CFG["password"])
            msg = MIMEText(content, "plain", "utf-8")
            msg["Subject"], msg["From"], msg["To"] = subject, SMTP_CFG["sender"], SMTP_CFG["receiver"]
            server.send_message(msg)
            print(f"✅ 邮件通过端口 {port} 发送成功！")
            return True  # 成功即退出
        except Exception as e:
            print(f"⚠ 端口 {port} 尝试失败: {str(e)}")
            if "SSL" in str(e):
                print("提示：端口465需使用SMTP_SSL，587需STARTTLS")
        finally:
            try: server.quit()
            except: pass
    
    # 重试逻辑（仅当全部失败时触发）
    for i in range(retries):
        print(f"⏳ 第{i+1}次重试（等待60秒）...")
        time.sleep(60)
        if send_mail(subject, content, retries=0): return True
    return False

if __name__ == "__main__":
    send_mail("【测试】邮件服务", "您的配置正常！")