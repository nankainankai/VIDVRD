from __future__ import annotations

"""全自动标注流程的中文 Prompt 模板。"""

import json
from typing import Any, Dict, List


def keyframe_screen_prompt(frame_results: List[Dict[str, Any]]) -> str:
    return (
        "你是视频关系标注质检助手。请根据关键帧检测统计判断视频是否值得继续标注。\n"
        "只输出 JSON，格式为：{\"decision\":\"keep|drop|crop\",\"reason\":\"中文原因\",\"crop_suggestion\":[x1,y1,x2,y2] 或 null}。\n"
        "判断原则：目标过少或画面无有效主体则 drop；主体集中在局部且裁剪能提升质量则 crop；否则 keep。\n"
        f"关键帧统计：{json.dumps(frame_results, ensure_ascii=False)}"
    )


def track_qc_prompt(risk_items: List[Dict[str, Any]]) -> str:
    return (
        "你是多目标追踪质检助手。请判断风险轨迹是否同一物体、框是否严重漂移、类别是否正确。\n"
        "只输出 JSON，格式为：{\"items\":[{\"track_id\":1,\"same_object\":true,\"bbox_ok\":true,\"class_ok\":true,\"action\":\"keep|review|drop\",\"reason\":\"中文原因\"}]}。\n"
        f"规则风险项：{json.dumps(risk_items, ensure_ascii=False)}"
    )


def global_relation_prompt(relations: List[Dict[str, Any]]) -> str:
    return (
        "你是视频级关系复核助手。请根据多个片段的候选关系判断是否存在跨片段持续关系和动态关系。\n"
        "只输出 JSON，格式为：{\"relations\":[{\"subject_track_id\":1,\"predicate\":\"follow\",\"object_track_id\":2,\"start_frame\":0,\"end_frame\":90,\"confidence\":0.8,\"reason\":\"中文证据\"}]}。\n"
        f"片段候选关系：{json.dumps(relations, ensure_ascii=False)}"
    )


def relation_verify_prompt(relations: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> str:
    return (
        "你是视频关系最终复核助手。请根据冲突、低置信度和候选关系给出最终动作。\n"
        "只输出 JSON，格式为：{\"actions\":[{\"action\":\"keep|delete|change_predicate|adjust_span|add_coupling\",\"index\":0,\"reason\":\"中文原因\"}]}。\n"
        f"候选关系：{json.dumps(relations, ensure_ascii=False)}\n"
        f"风险问题：{json.dumps(issues, ensure_ascii=False)}"
    )
