from ...tcm_states import TCMAgentState


def _attachment_value(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _attachment_status(item) -> str:
    status = _attachment_value(item, "status")
    return str(getattr(status, "value", status) or "")


async def handle_image_query(state: TCMAgentState) -> dict:
    """如实区分“未上传图片”和“已上传但视觉分析失败”。"""
    if state.report_analysis:
        analysis = (
            state.report_analysis.model_dump()
            if hasattr(state.report_analysis, "model_dump")
            else state.report_analysis
        )
        return {
            "answer": (
                f"已读取这份{analysis.get('report_type') or '医疗报告'}："
                f"{analysis.get('summary') or '已完成结构化提取。'}"
                "报告结果只能辅助理解，仍需结合症状、病史、查体和医生意见。"
            ),
            "steps": ["报告意图: 使用已有结构化解析结果"],
        }

    if state.tongue_analysis:
        analysis = state.tongue_analysis.model_dump()
        return {
            "answer": (
                "已收到舌像分析结果："
                f"舌色{analysis.get('tongue_color') or '未识别'}，"
                f"舌形{analysis.get('tongue_shape') or '未识别'}，"
                f"苔色{analysis.get('coating_color') or '未识别'}。"
                "舌象只能作为四诊合参的一部分，还需要结合症状与脉象判断。"
            ),
            "steps": ["图像意图: 使用已有舌像分析结果"],
        }

    if state.attachments:
        has_report = any(
            str(getattr(_attachment_value(item, "kind"), "value", _attachment_value(item, "kind")))
            == "medical_report"
            for item in state.attachments
        )
        failed = [
            item for item in state.attachments
            if _attachment_value(item, "analysis_error")
            or _attachment_status(item) == "analysis_failed"
        ]
        reason = _attachment_value(failed[0], "analysis_error") if failed else None
        if has_report:
            return {
                "answer": (
                    "已收到您上传的医疗报告，但本次解析未成功"
                    f"（{reason}）" if reason else "已收到您上传的医疗报告，但本次解析未成功。"
                ) + "我不会基于未读取的报告作判断。请确认文件未加密、页数和大小符合限制，必要时重新上传清晰图片或 PDF。",
                "steps": ["报告意图: 已收到报告但解析失败"],
            }
        return {
            "answer": (
                "已收到您上传的图片，但本次视觉分析未成功"
                f"（{reason}）" if reason else "已收到您上传的图片，但本次视觉分析未成功。"
            ) + "请确认当前选择的是支持图片输入的模型，或重新上传清晰舌照；也可以先用文字描述舌色、舌形和舌苔。",
            "steps": ["图像意图: 已收到图片但分析失败"],
        }

    return {
        "answer": (
            "当前聊天消息中没有收到可分析的图片。请先通过舌诊上传入口提交清晰的舌头照片，"
            "或用文字描述舌色、舌形和舌苔；不要仅凭舌照自行用药。"
        ),
        "steps": ["图像意图: 未收到图像，明确提示上传"],
    }
