"""
Generate a sample customer support knowledge base PDF.
Run: python generate_knowledge_base.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
import os

def create_pdf():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "customer_support_guide.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    s = getSampleStyleSheet()
    ts = ParagraphStyle('T', parent=s['Title'], fontSize=24, spaceAfter=30, textColor=HexColor('#2C3E50'), alignment=TA_CENTER)
    hs = ParagraphStyle('H', parent=s['Heading1'], fontSize=16, spaceAfter=12, spaceBefore=20, textColor=HexColor('#2C3E50'))
    sh = ParagraphStyle('SH', parent=s['Heading2'], fontSize=13, spaceAfter=8, spaceBefore=14, textColor=HexColor('#34495E'))
    bs = ParagraphStyle('B', parent=s['Normal'], fontSize=10.5, spaceAfter=8, leading=14)
    qs = ParagraphStyle('Q', parent=s['Normal'], fontSize=11, spaceAfter=4, textColor=HexColor('#2980B9'), fontName='Helvetica-Bold')
    ans = ParagraphStyle('A', parent=s['Normal'], fontSize=10.5, spaceAfter=12, leading=14, leftIndent=20)
    el = []
    hr = lambda: HRFlowable(width="100%", thickness=1, color=HexColor('#BDC3C7'))

    # Title
    el += [Spacer(1,2*inch), Paragraph("TechFlow Solutions", ts), Spacer(1,0.3*inch)]
    el += [Paragraph("Customer Support Knowledge Base", ParagraphStyle('ST', parent=s['Title'], fontSize=16, textColor=HexColor('#7F8C8D'), alignment=TA_CENTER))]
    el += [Spacer(1,0.5*inch), Paragraph("Version 2.1 | April 2026", ParagraphStyle('V', parent=s['Normal'], fontSize=10, textColor=HexColor('#95A5A6'), alignment=TA_CENTER))]
    el += [PageBreak()]

    # 1. Company
    el += [Paragraph("1. Company Overview & Products", hs), hr(), Spacer(1,0.2*inch)]
    el += [Paragraph("TechFlow Solutions is a leading provider of cloud-based business management software. Founded in 2018, we serve over 50,000 businesses worldwide with our suite of productivity and communication tools.", bs)]
    for p in [
        "<b>TechFlow Workspace</b> - Cloud collaboration platform. Basic ($9.99/mo), Professional ($24.99/mo), Enterprise ($49.99/mo).",
        "<b>TechFlow CRM</b> - Customer relationship management with lead tracking. Starts at $19.99/mo per user.",
        "<b>TechFlow Analytics</b> - Business intelligence and dashboards. Add-on for $14.99/mo.",
        "<b>TechFlow Connect</b> - Communication tool. Free for up to 10 users, then $7.99/user/mo.",
    ]:
        el += [Paragraph(f"• {p}", bs)]

    # 2. Account Management
    el += [Paragraph("2. Account Management", hs), hr()]
    el += [Paragraph("2.1 Creating an Account", sh)]
    el += [Paragraph("Visit app.techflow.com/signup. Enter business email. Choose a strong password (min 12 chars with uppercase, lowercase, number, special char). Verify via confirmation link (sent within 5 min). Complete profile. All new accounts get a 14-day free trial of the Professional plan.", bs)]
    el += [Paragraph("2.2 Password Reset", sh)]
    el += [Paragraph("Go to app.techflow.com/reset-password. Enter registered email. Click 'Send Reset Link'. Check email (including spam) - link expires after 30 minutes. Create new password. If no email in 10 min, contact support@techflow.com. We cannot send passwords via phone or chat for security.", bs)]
    el += [Paragraph("2.3 Account Deactivation & Deletion", sh)]
    el += [Paragraph("Deactivate: Settings > Account > Deactivation. Data retained 90 days; reactivate by logging in. Permanent deletion: submit request at support.techflow.com/delete-account. Irreversible, takes 30 days. All data removed. Cancel active subscriptions first.", bs)]
    el += [Paragraph("2.4 Multi-Factor Authentication", sh)]
    el += [Paragraph("Enable in Settings > Security > MFA. Supports Google Authenticator, Authy, SMS, and YubiKey. Enterprise plans include mandatory MFA enforcement.", bs)]

    # 3. Billing
    el += [PageBreak(), Paragraph("3. Billing & Payments", hs), hr()]
    el += [Paragraph("3.1 Subscription Plans", sh)]
    td = [['Plan','Monthly','Annual','Features'],
          ['Basic','$9.99/user','$99.99/user','5GB storage, basic support, 3 projects'],
          ['Professional','$24.99/user','$249.99/user','50GB storage, priority support, unlimited projects'],
          ['Enterprise','$49.99/user','$499.99/user','Unlimited storage, 24/7 support, custom integrations']]
    t = Table(td, colWidths=[80,90,90,200])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#2C3E50')),('TEXTCOLOR',(0,0),(-1,0),HexColor('#FFFFFF')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.5,HexColor('#BDC3C7')),
        ('ALIGN',(1,0),(2,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('PADDING',(0,0),(-1,-1),6)]))
    el += [t, Spacer(1,0.2*inch)]
    el += [Paragraph("3.2 Payment Methods", sh)]
    el += [Paragraph("Accepted: Visa, Mastercard, Amex, Discover, PayPal, bank transfers (ACH for annual Enterprise), wire transfers for invoices over $5,000. All payments via Stripe with PCI-DSS Level 1 compliance.", bs)]
    el += [Paragraph("3.3 Refund Policy", sh)]
    el += [Paragraph("30-day money-back guarantee on new subscriptions. Request within 30 days. Processed in 5-10 business days. Annual plans prorated. No refunds for: partial months, add-ons used 14+ days, or ToS violations. Email billing@techflow.com with account ID.", bs)]
    el += [Paragraph("3.4 Invoice Management", sh)]
    el += [Paragraph("Invoices generated on 1st of each month. Access: Settings > Billing > Invoice History. PDF download available. Update billing email in Settings > Billing > Billing Contact. Report discrepancies within 60 days.", bs)]

    # 4. Technical Support
    el += [PageBreak(), Paragraph("4. Technical Support & Troubleshooting", hs), hr()]
    el += [Paragraph("4.1 System Requirements", sh)]
    el += [Paragraph("Chrome 90+, Firefox 88+, Safari 14+, or Edge 90+. Min 4GB RAM, 5 Mbps internet. Mobile: iOS 15+, Android 11+. Desktop: Windows 10/11, macOS 12+.", bs)]
    el += [Paragraph("4.2 Common Issues", sh)]
    for title, sol in [
        ("Login Issues", "Clear cache/cookies. Try incognito. Verify email. Check if deactivated. For SSO contact IT admin. Reset at app.techflow.com/reset-password."),
        ("Slow Performance", "Check internet speed (min 5 Mbps). Clear cache. Disable interfering extensions. Try different browser. Check status.techflow.com."),
        ("File Upload Failures", "Max size: 500MB Professional, 2GB Enterprise. Supported: PDF, DOCX, XLSX, PPTX, JPG, PNG, MP4, ZIP. Check storage quota in Settings > Storage."),
        ("Video Conferencing", "Grant camera/mic permissions. Close other apps using camera. Min 2 Mbps for video, 5 Mbps HD. Test at app.techflow.com/test-connection. Try disconnecting VPN."),
        ("Email Integration", "Verify provider supported (Gmail, Outlook, Yahoo, IMAP). Re-authorize in Settings > Integrations > Email. Enable IMAP in email provider."),
        ("Data Sync Issues", "Same account on all devices. Enable auto-sync in Settings > Sync. Force manual sync via icon. Log out/in on affected device."),
    ]:
        el += [Paragraph(f"Issue: {title}", qs), Paragraph(f"Solution: {sol}", ans)]
    el += [Paragraph("4.3 API Issues", sh)]
    el += [Paragraph("Docs at docs.techflow.com/api. Rate limits: Basic 100/min, Professional 500/min, Enterprise 2000/min. 429 error = rate exceeded, use exponential backoff. Webhook endpoints must return 200 within 10s. Regenerate API keys in Settings > Developer.", bs)]

    # 5. Features
    el += [PageBreak(), Paragraph("5. Product Features & How-To Guides", hs), hr()]
    el += [Paragraph("5.1 Team Workspaces", sh)]
    el += [Paragraph("Click '+ New Workspace'. Name and set visibility. Invite via email/link. Permissions: Admin (full), Editor (create/edit), Viewer (read-only). Up to 500 members Professional, unlimited Enterprise.", bs)]
    el += [Paragraph("5.2 Project Management", sh)]
    el += [Paragraph("Kanban boards, Gantt charts, list views. Assign tasks with due dates, priorities (Low/Medium/High/Critical). Automated workflows available. Built-in time tracking. Reports in Settings > Projects > Reports.", bs)]
    el += [Paragraph("5.3 File Sharing", sh)]
    el += [Paragraph("Drag files into channels/DMs. Real-time co-editing for docs, sheets, presentations. Version history: 30 versions Professional, unlimited Enterprise. Right-click > Version History. File-level permissions.", bs)]

    # 6. Security
    el += [Paragraph("6. Security & Privacy", hs), hr()]
    el += [Paragraph("SOC 2 Type II, ISO 27001, GDPR compliant. AES-256 encryption at rest, TLS 1.3 in transit. AWS hosted, 99.99% uptime. Regular penetration testing. Enterprise: dedicated infrastructure, custom data residency. Report vulnerabilities to security@techflow.com.", bs)]
    el += [Paragraph("Data Retention: Active accounts indefinite. Cancelled 90 days. Deleted 30 days. Audit logs 1 year (Enterprise 3). Daily backups 30 days.", bs)]

    # 7. SLA
    el += [PageBreak(), Paragraph("7. Service Level Agreements", hs), hr()]
    sd = [['Plan','Uptime','Response','Hours'],['Basic','99.5%','24h','Mon-Fri 9-5 EST'],
          ['Professional','99.9%','4h','Mon-Sat 8-8 EST'],['Enterprise','99.99%','1h','24/7/365']]
    st2 = Table(sd, colWidths=[80,80,80,220])
    st2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#2C3E50')),('TEXTCOLOR',(0,0),(-1,0),HexColor('#FFFFFF')),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.5,HexColor('#BDC3C7')),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('PADDING',(0,0),(-1,-1),6)]))
    el += [st2, Spacer(1,0.2*inch)]
    el += [Paragraph("SLA credits: 99.0-99.5% = 10% credit, 98.0-99.0% = 25%, below 98% = 50%. Credits applied next billing cycle, request within 30 days. Scheduled maintenance (72h notice) excluded.", bs)]

    # 8. Escalation
    el += [Paragraph("8. Escalation Procedures", hs), hr()]
    el += [Paragraph("Escalate when: Customer waiting beyond SLA, data loss/security breach, significant dissatisfaction, requires elevated permissions, billing disputes over $500.", bs)]
    el += [Paragraph("Levels: L1 (Tier 1) - common issues, 80% resolution target. L2 (Tier 2) - complex/billing, triggered after 2 L1 attempts. L3 (Engineering) - bugs/infra. L4 (Management) - SLA/data breaches, P1 incidents.", bs)]

    # 9. FAQ
    el += [PageBreak(), Paragraph("9. Frequently Asked Questions", hs), hr()]
    for q, a in [
        ("How do I upgrade my plan?", "Settings > Billing > Change Plan. Select new plan. Upgrades immediate, prorated charge."),
        ("Can I downgrade?", "Yes. Settings > Billing > Change Plan. Takes effect end of cycle. May lose exclusive features. Excess data retained 30 days."),
        ("How to export data?", "Settings > Account > Data Export. Choose categories. ZIP file, up to 24h for large accounts. Email notification when ready."),
        ("Free plan available?", "TechFlow Connect free for up to 10 users. Workspace: 14-day free Professional trial, no credit card needed."),
        ("How to cancel?", "Settings > Billing > Cancel Subscription. Active until end of billing period. Data retained 90 days. Reactivate anytime during retention."),
        ("Educational/nonprofit discounts?", "50% off for verified educational institutions, 30% off for nonprofits. Apply at techflow.com/discounts. Processing: 5-7 business days."),
        ("Exceeded storage limit?", "Warnings at 80% and 90%. At 100% no new uploads but existing files accessible. Upgrade plan or add storage at $2.99/10GB/mo."),
        ("How to contact support?", "Email: support@techflow.com. Live Chat: in-app during support hours. Phone: 1-800-TECHFLOW (Enterprise only). Help Center: help.techflow.com."),
    ]:
        el += [Paragraph(f"Q: {q}", qs), Paragraph(f"A: {a}", ans)]

    # 10. Contact
    el += [Paragraph("10. Contact Information", hs), hr()]
    el += [Paragraph("Support: support@techflow.com | Billing: billing@techflow.com | Security: security@techflow.com | Sales: sales@techflow.com | Phone: 1-800-TECHFLOW | Help: help.techflow.com | Status: status.techflow.com | Office: 123 Innovation Drive, Suite 400, San Francisco, CA 94105", bs)]

    doc.build(el)
    print(f"[OK] PDF created: {path}")
    return path

if __name__ == "__main__":
    create_pdf()
