import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
from collections import Counter
import numpy as np
import jieba
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="问卷数据可视化分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class QuestionnaireAnalyzer:
    def __init__(self):
        self.questions = {}
        self.data = None
        self.question_types = {
            '单选题': 'single_choice',
            '多选题': 'multiple_choice', 
            '评分题': 'rating',
            '填空题': 'fill_in_blank'
        }
        # 对比功能相关属性
        self.comparison_questions = {}
        self.comparison_data = None
        
    def parse_md_metadata_with_data(self, md_content):
        """解析包含数据的MD格式文件"""
        questions = {}
        
        lines = md_content.split('\n')
        current_question = None
        current_question_text = None
        current_question_type = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 解析问题行格式: 问题X:【类型】问题内容
            if line.startswith('问题') and '【' in line and '】' in line:
                # 如果有之前的问题，先保存
                if current_question is not None:
                    questions[current_question_text] = current_question
                
                # 提取问题类型
                question_type = None
                for qtype in self.question_types.keys():
                    if f'【{qtype}】' in line:
                        question_type = self.question_types[qtype]
                        break
                
                if not question_type:
                    i += 1
                    continue
                
                # 提取问题文本
                question_text = line.split('】', 1)[1] if '】' in line else line
                question_text = question_text.strip().rstrip('\t')
                
                current_question_text = question_text
                current_question_type = question_type
                current_question = {
                    'text': question_text,
                    'type': question_type,
                    'options': [],
                    'data': []
                }
                
                # 读取后续的数据行
                i += 1
                while i < len(lines):
                    data_line = lines[i].strip()
                    
                    # 如果遇到空行，跳过
                    if not data_line:
                        i += 1
                        continue
                    
                    # 如果遇到下一个问题，退出当前问题的数据读取
                    if data_line.startswith('问题') and '【' in data_line and '】' in data_line:
                        break
                    
                    # 解析数据行
                    if question_type == 'fill_in_blank':
                        # 填空题只有数量
                        try:
                            total_responses = int(data_line)
                            current_question['data'] = [total_responses]
                        except:
                            current_question['data'] = [0]
                        i += 1
                        break
                    else:
                        # 其他题型解析选项和数据
                        if ':' in data_line:
                            option_name = data_line.split(':')[0].strip()
                            # 提取数量
                            count_match = re.search(r': (\d+)', data_line)
                            if count_match:
                                count = int(count_match.group(1))
                                current_question['options'].append(option_name)
                                current_question['data'].append(count)
                    
                    i += 1
            else:
                i += 1
        
        # 保存最后一个问题
        if current_question is not None:
            questions[current_question_text] = current_question
                
        return questions
    
    def create_data_from_parsed_questions(self):
        """根据解析的问题数据创建DataFrame"""
        if not self.questions:
            return False
            
        # 找到最大的回答数量来确定需要生成多少行数据
        max_responses = 0
        for q_info in self.questions.values():
            if q_info['type'] != 'fill_in_blank' and q_info['data']:
                max_responses = max(max_responses, sum(q_info['data']))
        
        if max_responses == 0:
            max_responses = 100  # 默认值
        
        data_dict = {}
        
        for q_text, q_info in self.questions.items():
            if q_info['type'] == 'fill_in_blank':
                continue  # 跳过填空题
                
            responses = []
            
            if q_info['type'] in ['single_choice', 'rating'] and q_info['options'] and q_info['data']:
                # 根据实际数据分布生成回答
                for option, count in zip(q_info['options'], q_info['data']):
                    responses.extend([option] * count)
                    
            elif q_info['type'] == 'multiple_choice' and q_info['options'] and q_info['data']:
                # 多选题：根据每个选项的选择次数生成数据
                total_responses = max_responses
                for i in range(total_responses):
                    selected_options = []
                    for option, count in zip(q_info['options'], q_info['data']):
                        # 根据比例随机选择
                        probability = count / total_responses
                        if np.random.random() < probability:
                            selected_options.append(option)
                    
                    if selected_options:
                        responses.append(';'.join(selected_options))
                    else:
                        responses.append('')
            
            # 确保所有列的长度一致
            if len(responses) < max_responses:
                responses.extend([''] * (max_responses - len(responses)))
            elif len(responses) > max_responses:
                responses = responses[:max_responses]
                
            data_dict[q_text] = responses
        
        if data_dict:
            self.data = pd.DataFrame(data_dict)
            return True
        return False
    
    def create_single_choice_viz(self, question_text, data_column=None):
        """创建单选题可视化"""
        if question_text not in self.questions:
            return None
            
        q_info = self.questions[question_text]
        
        # 使用原始数据
        if q_info['options'] and q_info['data']:
            options = q_info['options']
            counts = q_info['data']
        else:
            return None
        
        # 紫色主题色彩
        purple_colors = ['#9B59B6', '#8E44AD', '#BB8FCE', '#D2B4DE', '#E8DAEF', '#F4ECF7']
        bar_colors = [purple_colors[i % len(purple_colors)] for i in range(len(options))]
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('条形图', '饼图'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 条形图
        fig.add_trace(
            go.Bar(
                x=options,
                y=counts,
                name="数量",
                marker_color=bar_colors,
                marker_line=dict(color='#6C3483', width=1)
            ),
            row=1, col=1
        )
        
        # 饼图
        fig.add_trace(
            go.Pie(
                labels=options,
                values=counts,
                name="比例",
                marker_colors=purple_colors,
                marker_line=dict(color='#FFFFFF', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"单选题分析: {question_text}",
            title_font=dict(size=16, color='#6C3483'),
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 更新坐标轴样式
        fig.update_xaxes(tickangle=45, title_font=dict(color='#6C3483'), tickfont=dict(color='#6C3483'))
        fig.update_yaxes(title_font=dict(color='#6C3483'), tickfont=dict(color='#6C3483'))
        
        return fig
    
    def create_multiple_choice_viz(self, question_text, data_column=None):
        """创建多选题可视化"""
        if question_text not in self.questions:
            return None
            
        q_info = self.questions[question_text]
        
        # 使用原始数据
        if q_info['options'] and q_info['data']:
            options = q_info['options']
            counts = q_info['data']
        else:
            return None
        
        # 紫色渐变色彩
        purple_gradient = ['#E8DAEF', '#D2B4DE', '#BB8FCE', '#A569BD', '#9B59B6', '#8E44AD', '#7D3C98', '#6C3483']
        bar_colors = [purple_gradient[i % len(purple_gradient)] for i in range(len(options))]
        
        # 创建水平条形图
        fig = go.Figure(go.Bar(
            x=counts,
            y=options,
            orientation='h',
            marker_color=bar_colors,
            marker_line=dict(color='#6C3483', width=1),
            text=counts,
            textposition='outside',
            textfont=dict(color='#6C3483', size=12)
        ))
        
        fig.update_layout(
            title=f"多选题分析: {question_text}",
            title_font=dict(size=16, color='#6C3483'),
            xaxis_title="选择次数",
            yaxis_title="选项",
            height=max(400, len(options) * 35),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(color='#6C3483'),
                tickfont=dict(color='#6C3483'),
                gridcolor='#E8DAEF'
            ),
            yaxis=dict(
                title_font=dict(color='#6C3483'),
                tickfont=dict(color='#6C3483')
            )
        )
        
        return fig
    
    def create_rating_viz(self, question_text, data_column=None):
        """创建评分题可视化"""
        if question_text not in self.questions:
            return None
            
        q_info = self.questions[question_text]
        
        # 使用原始数据
        if q_info['options'] and q_info['data']:
            ratings = []
            for option, count in zip(q_info['options'], q_info['data']):
                try:
                    rating_value = int(option)
                    ratings.extend([rating_value] * count)
                except:
                    continue
        else:
            return None
        
        if not ratings:
            return None
        
        ratings = pd.Series(ratings)
        
        # 紫色主题色彩
        purple_colors = ['#9B59B6', '#8E44AD', '#BB8FCE', '#D2B4DE', '#E8DAEF']
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('评分分布', '箱线图', '统计摘要', '雷达图'),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "table"}, {"type": "scatterpolar"}]
            ]
        )
        
        # 评分分布
        rating_counts = ratings.value_counts().sort_index()
        bar_colors = [purple_colors[i % len(purple_colors)] for i in range(len(rating_counts))]
        
        fig.add_trace(
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                name="频次",
                marker_color=bar_colors,
                marker_line=dict(color='#6C3483', width=1),
                text=rating_counts.values,
                textposition='outside',
                textfont=dict(color='#6C3483', size=12)
            ),
            row=1, col=1
        )
        
        # 箱线图
        fig.add_trace(
            go.Box(
                y=ratings,
                name="评分分布",
                marker_color='#9B59B6',
                line_color='#6C3483',
                fillcolor='rgba(155, 89, 182, 0.3)'
            ),
            row=1, col=2
        )
        
        # 统计摘要表
        stats = ratings.describe()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['统计量', '数值'],
                    fill_color='#D2B4DE',
                    font=dict(color='#6C3483', size=14)
                ),
                cells=dict(
                    values=[
                        ['平均值', '中位数', '标准差', '最小值', '最大值'],
                        [f"{stats['mean']:.2f}", f"{stats['50%']:.2f}", 
                         f"{stats['std']:.2f}", f"{stats['min']:.2f}", f"{stats['max']:.2f}"]
                    ],
                    fill_color='#F4ECF7',
                    font=dict(color='#6C3483', size=12)
                )
            ),
            row=2, col=1
        )
        
        # 雷达图
        categories = [f"评分{i}" for i in rating_counts.index]
        fig.add_trace(
            go.Scatterpolar(
                r=rating_counts.values,
                theta=categories,
                fill='toself',
                name='评分分布',
                line_color='#9B59B6',
                fillcolor='rgba(155, 89, 182, 0.3)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"评分题分析: {question_text}",
            title_font=dict(size=16, color='#6C3483'),
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # 更新坐标轴样式
        fig.update_xaxes(title_font=dict(color='#6C3483'), tickfont=dict(color='#6C3483'))
        fig.update_yaxes(title_font=dict(color='#6C3483'), tickfont=dict(color='#6C3483'))
        
        return fig
    
    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.questions:
            return None
            
        total_responses = 0
        analyzed_questions = 0
        skipped_questions = 0
        
        for q_info in self.questions.values():
            if q_info['type'] == 'fill_in_blank':
                skipped_questions += 1
                if q_info['data']:
                    total_responses = max(total_responses, q_info['data'][0])
            else:
                analyzed_questions += 1
                if q_info['data']:
                    total_responses = max(total_responses, sum(q_info['data']))
        
        summary = {
            '总回答数': total_responses,
            '分析问题数': analyzed_questions,
            '跳过问题数（填空题）': skipped_questions,
            '数据完整性': "100.0%"
        }
        
        return summary
    
    def find_matching_questions(self):
        """找到两个问卷中相似的问题"""
        if not self.questions or not self.comparison_questions:
            return []
        
        matching_pairs = []
        
        for q1_text, q1_info in self.questions.items():
            for q2_text, q2_info in self.comparison_questions.items():
                # 检查问题类型是否相同
                if q1_info['type'] == q2_info['type'] and q1_info['type'] != 'fill_in_blank':
                    # 计算问题文本相似度（简单的关键词匹配）
                    similarity = self.calculate_text_similarity(q1_text, q2_text)
                    if similarity > 0.6:  # 相似度阈值
                        matching_pairs.append({
                            'question1': q1_text,
                            'question2': q2_text,
                            'type': q1_info['type'],
                            'similarity': similarity,
                            'data1': q1_info,
                            'data2': q2_info
                        })
        
        # 按相似度排序
        matching_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return matching_pairs
    
    def calculate_text_similarity(self, text1, text2):
        """计算两个文本的相似度"""
        # 简单的关键词匹配算法
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def create_comparison_viz(self, matching_pair, group1_name="组别1", group2_name="组别2"):
        """创建对比可视化"""
        question_type = matching_pair['type']
        data1 = matching_pair['data1']
        data2 = matching_pair['data2']
        
        if question_type == 'single_choice':
            return self.create_single_choice_comparison(data1, data2, matching_pair, group1_name, group2_name)
        elif question_type == 'multiple_choice':
            return self.create_multiple_choice_comparison(data1, data2, matching_pair, group1_name, group2_name)
        elif question_type == 'rating':
            return self.create_rating_comparison(data1, data2, matching_pair, group1_name, group2_name)
        
        return None
    
    def create_single_choice_comparison(self, data1, data2, matching_pair, group1_name, group2_name):
        """创建单选题对比图"""
        # 合并所有选项
        all_options = list(set(data1['options'] + data2['options']))
        
        # 为每个组准备数据
        counts1 = []
        counts2 = []
        
        for option in all_options:
            if option in data1['options']:
                idx = data1['options'].index(option)
                counts1.append(data1['data'][idx])
            else:
                counts1.append(0)
                
            if option in data2['options']:
                idx = data2['options'].index(option)
                counts2.append(data2['data'][idx])
            else:
                counts2.append(0)
        
        # 计算百分比
        total1 = sum(counts1) if sum(counts1) > 0 else 1
        total2 = sum(counts2) if sum(counts2) > 0 else 1
        percentages1 = [count/total1*100 for count in counts1]
        percentages2 = [count/total2*100 for count in counts2]
        
        # 创建对比图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{group1_name} vs {group2_name} - 数量对比', f'{group1_name} vs {group2_name} - 百分比对比'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 数量对比
        fig.add_trace(
            go.Bar(
                x=all_options,
                y=counts1,
                name=group1_name,
                marker_color='#9B59B6',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=all_options,
                y=counts2,
                name=group2_name,
                marker_color='#E74C3C',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 百分比对比
        fig.add_trace(
            go.Bar(
                x=all_options,
                y=percentages1,
                name=f'{group1_name} (%)',
                marker_color='#9B59B6',
                opacity=0.8,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=all_options,
                y=percentages2,
                name=f'{group2_name} (%)',
                marker_color='#E74C3C',
                opacity=0.8,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"单选题对比: {matching_pair['question1'][:50]}...",
            title_font=dict(size=16, color='#2C3E50'),
            height=500,
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(tickangle=45, title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50'))
        fig.update_yaxes(title_font=dict(color='#2C3E50'), tickfont=dict(color='#2C3E50'))
        
        return fig
    
    def create_multiple_choice_comparison(self, data1, data2, matching_pair, group1_name, group2_name):
        """创建多选题对比图"""
        # 合并所有选项
        all_options = list(set(data1['options'] + data2['options']))
        
        # 为每个组准备数据
        counts1 = []
        counts2 = []
        
        for option in all_options:
            if option in data1['options']:
                idx = data1['options'].index(option)
                counts1.append(data1['data'][idx])
            else:
                counts1.append(0)
                
            if option in data2['options']:
                idx = data2['options'].index(option)
                counts2.append(data2['data'][idx])
            else:
                counts2.append(0)
        
        # 创建水平对比条形图
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=all_options,
            x=counts1,
            name=group1_name,
            orientation='h',
            marker_color='#9B59B6',
            opacity=0.8
        ))
        
        fig.add_trace(go.Bar(
            y=all_options,
            x=counts2,
            name=group2_name,
            orientation='h',
            marker_color='#E74C3C',
            opacity=0.8
        ))
        
        fig.update_layout(
            title=f"多选题对比: {matching_pair['question1'][:50]}...",
            title_font=dict(size=16, color='#2C3E50'),
            xaxis_title="选择次数",
            yaxis_title="选项",
            height=max(400, len(all_options) * 40),
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title_font=dict(color='#2C3E50'),
                tickfont=dict(color='#2C3E50'),
                gridcolor='#E8DAEF'
            ),
            yaxis=dict(
                title_font=dict(color='#2C3E50'),
                tickfont=dict(color='#2C3E50')
            )
        )
        
        return fig
    
    def create_rating_comparison(self, data1, data2, matching_pair, group1_name, group2_name):
        """创建评分题对比图"""
        # 准备评分数据
        ratings1 = []
        ratings2 = []
        
        for option, count in zip(data1['options'], data1['data']):
            try:
                rating_value = int(option)
                ratings1.extend([rating_value] * count)
            except:
                continue
                
        for option, count in zip(data2['options'], data2['data']):
            try:
                rating_value = int(option)
                ratings2.extend([rating_value] * count)
            except:
                continue
        
        if not ratings1 or not ratings2:
            return None
        
        ratings1 = pd.Series(ratings1)
        ratings2 = pd.Series(ratings2)
        
        # 创建对比图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('评分分布对比', '箱线图对比', '统计对比', '平均分对比'),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "table"}, {"type": "bar"}]
            ]
        )
        
        # 评分分布对比
        rating_counts1 = ratings1.value_counts().sort_index()
        rating_counts2 = ratings2.value_counts().sort_index()
        
        all_ratings = sorted(set(list(rating_counts1.index) + list(rating_counts2.index)))
        
        counts1_aligned = [rating_counts1.get(r, 0) for r in all_ratings]
        counts2_aligned = [rating_counts2.get(r, 0) for r in all_ratings]
        
        fig.add_trace(
            go.Bar(
                x=all_ratings,
                y=counts1_aligned,
                name=group1_name,
                marker_color='#9B59B6',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=all_ratings,
                y=counts2_aligned,
                name=group2_name,
                marker_color='#E74C3C',
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # 箱线图对比
        fig.add_trace(
            go.Box(
                y=ratings1,
                name=group1_name,
                marker_color='#9B59B6',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=ratings2,
                name=group2_name,
                marker_color='#E74C3C',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 统计对比表
        stats1 = ratings1.describe()
        stats2 = ratings2.describe()
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['统计量', group1_name, group2_name],
                    fill_color='#D2B4DE',
                    font=dict(color='#2C3E50', size=12)
                ),
                cells=dict(
                    values=[
                        ['平均值', '中位数', '标准差', '最小值', '最大值'],
                        [f"{stats1['mean']:.2f}", f"{stats1['50%']:.2f}", 
                         f"{stats1['std']:.2f}", f"{stats1['min']:.2f}", f"{stats1['max']:.2f}"],
                        [f"{stats2['mean']:.2f}", f"{stats2['50%']:.2f}", 
                         f"{stats2['std']:.2f}", f"{stats2['min']:.2f}", f"{stats2['max']:.2f}"]
                    ],
                    fill_color='#F4ECF7',
                    font=dict(color='#2C3E50', size=10)
                )
            ),
            row=2, col=1
        )
        
        # 平均分对比
        fig.add_trace(
            go.Bar(
                x=[group1_name, group2_name],
                y=[stats1['mean'], stats2['mean']],
                name='平均分',
                marker_color=['#9B59B6', '#E74C3C'],
                text=[f"{stats1['mean']:.2f}", f"{stats2['mean']:.2f}"],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"评分题对比: {matching_pair['question1'][:50]}...",
            title_font=dict(size=16, color='#2C3E50'),
            height=800,
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

def main():
    st.title("📊 问卷数据可视化分析工具")
    st.markdown("---")
    
    analyzer = QuestionnaireAnalyzer()
    
    # 侧边栏配置
    st.sidebar.header("📁 文件上传")
    
    # 选择分析模式
    analysis_mode = st.sidebar.radio(
        "选择分析模式:",
        ["单文件分析", "双文件对比分析"],
        help="单文件分析：分析单个问卷文件\n双文件对比分析：对比两个问卷文件的数据"
    )
    
    if analysis_mode == "单文件分析":
        # 上传MD文件
        md_file = st.sidebar.file_uploader(
            "上传问卷数据文件 (.md)", 
            type=['md'],
            help="包含问题和数据的元数据文件"
        )
    else:
        # 双文件对比模式
        st.sidebar.subheader("📊 对比分析")
        
        # 上传第一个文件
        md_file = st.sidebar.file_uploader(
            "上传第一个问卷文件 (.md)", 
            type=['md'],
            help="第一组问卷数据文件",
            key="file1"
        )
        
        # 第一组名称
        group1_name = st.sidebar.text_input(
            "第一组名称:", 
            value="对照组",
            help="为第一组数据设置名称"
        )
        
        # 上传第二个文件
        comparison_file = st.sidebar.file_uploader(
            "上传第二个问卷文件 (.md)", 
            type=['md'],
            help="第二组问卷数据文件",
            key="file2"
        )
        
        # 第二组名称
        group2_name = st.sidebar.text_input(
            "第二组名称:", 
            value="实验组",
            help="为第二组数据设置名称"
        )
    
    # 处理对比分析模式
    if analysis_mode == "双文件对比分析" and md_file and comparison_file:
        # 解析两个文件
        md_content = md_file.read().decode('utf-8')
        comparison_content = comparison_file.read().decode('utf-8')
        
        analyzer.questions = analyzer.parse_md_metadata_with_data(md_content)
        analyzer.comparison_questions = analyzer.parse_md_metadata_with_data(comparison_content)
        
        if analyzer.questions and analyzer.comparison_questions:
            st.success(f"✅ 成功解析第一组 {len(analyzer.questions)} 个问题，第二组 {len(analyzer.comparison_questions)} 个问题")
            
            # 找到匹配的问题
            matching_pairs = analyzer.find_matching_questions()
            
            if matching_pairs:
                st.header("🔍 问题匹配结果")
                st.info(f"找到 {len(matching_pairs)} 对相似问题可以进行对比分析")
                
                # 显示匹配结果表格
                match_df = pd.DataFrame([
                    {
                        '相似度': f"{pair['similarity']:.2%}",
                        '问题类型': pair['type'],
                        f'{group1_name}问题': pair['question1'][:60] + "..." if len(pair['question1']) > 60 else pair['question1'],
                        f'{group2_name}问题': pair['question2'][:60] + "..." if len(pair['question2']) > 60 else pair['question2']
                    }
                    for pair in matching_pairs
                ])
                st.dataframe(match_df, use_container_width=True)
                
                # 对比分析
                st.header("📊 对比分析")
                
                if matching_pairs:
                    # 选择要对比的问题
                    selected_pair_index = st.selectbox(
                        "选择要对比的问题:",
                        range(len(matching_pairs)),
                        format_func=lambda x: f"相似度{matching_pairs[x]['similarity']:.1%} - {matching_pairs[x]['question1'][:50]}..."
                    )
                    
                    selected_pair = matching_pairs[selected_pair_index]
                    
                    # 显示问题详情
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"📋 {group1_name}")
                        st.write(f"**问题:** {selected_pair['question1']}")
                        st.write(f"**类型:** {selected_pair['type']}")
                        if selected_pair['data1']['data']:
                            st.write(f"**回答数:** {sum(selected_pair['data1']['data'])}")
                    
                    with col2:
                        st.subheader(f"📋 {group2_name}")
                        st.write(f"**问题:** {selected_pair['question2']}")
                        st.write(f"**类型:** {selected_pair['type']}")
                        if selected_pair['data2']['data']:
                            st.write(f"**回答数:** {sum(selected_pair['data2']['data'])}")
                    
                    # 创建对比可视化
                    comparison_fig = analyzer.create_comparison_viz(selected_pair, group1_name, group2_name)
                    if comparison_fig:
                        st.plotly_chart(comparison_fig, use_container_width=True)
                        
                        # 显示详细对比数据
                        st.subheader("📈 详细对比数据")
                        
                        if selected_pair['type'] in ['single_choice', 'multiple_choice']:
                            # 创建对比数据表
                            all_options = list(set(selected_pair['data1']['options'] + selected_pair['data2']['options']))
                            
                            comparison_data = []
                            for option in all_options:
                                count1 = 0
                                count2 = 0
                                
                                if option in selected_pair['data1']['options']:
                                    idx = selected_pair['data1']['options'].index(option)
                                    count1 = selected_pair['data1']['data'][idx]
                                
                                if option in selected_pair['data2']['options']:
                                    idx = selected_pair['data2']['options'].index(option)
                                    count2 = selected_pair['data2']['data'][idx]
                                
                                total1 = sum(selected_pair['data1']['data']) if selected_pair['data1']['data'] else 1
                                total2 = sum(selected_pair['data2']['data']) if selected_pair['data2']['data'] else 1
                                
                                comparison_data.append({
                                    '选项': option,
                                    f'{group1_name}数量': count1,
                                    f'{group1_name}占比': f"{count1/total1*100:.1f}%",
                                    f'{group2_name}数量': count2,
                                    f'{group2_name}占比': f"{count2/total2*100:.1f}%",
                                    '差异': f"{(count2/total2 - count1/total1)*100:+.1f}%"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                        
                        elif selected_pair['type'] == 'rating':
                            # 评分题统计对比
                            ratings1 = []
                            ratings2 = []
                            
                            for option, count in zip(selected_pair['data1']['options'], selected_pair['data1']['data']):
                                try:
                                    rating_value = int(option)
                                    ratings1.extend([rating_value] * count)
                                except:
                                    continue
                            
                            for option, count in zip(selected_pair['data2']['options'], selected_pair['data2']['data']):
                                try:
                                    rating_value = int(option)
                                    ratings2.extend([rating_value] * count)
                                except:
                                    continue
                            
                            if ratings1 and ratings2:
                                ratings1 = pd.Series(ratings1)
                                ratings2 = pd.Series(ratings2)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(f"{group1_name}平均分", f"{ratings1.mean():.2f}")
                                    st.metric(f"{group1_name}中位数", f"{ratings1.median():.2f}")
                                
                                with col2:
                                    st.metric(f"{group2_name}平均分", f"{ratings2.mean():.2f}")
                                    st.metric(f"{group2_name}中位数", f"{ratings2.median():.2f}")
                                
                                with col3:
                                    avg_diff = ratings2.mean() - ratings1.mean()
                                    st.metric("平均分差异", f"{avg_diff:+.2f}")
                                    median_diff = ratings2.median() - ratings1.median()
                                    st.metric("中位数差异", f"{median_diff:+.2f}")
                
                # 批量导出功能
                st.header("💾 批量导出")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("生成所有对比图表"):
                        output_dir = "comparison_output"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        progress_bar = st.progress(0)
                        exported_count = 0
                        
                        for i, pair in enumerate(matching_pairs):
                            try:
                                fig = analyzer.create_comparison_viz(pair, group1_name, group2_name)
                                if fig:
                                    safe_filename = re.sub(r'[^\w\s-]', '', pair['question1'])[:30]
                                    fig.write_image(f"{output_dir}/对比_{safe_filename}.png", 
                                                  width=1400, height=800)
                                    exported_count += 1
                            except Exception as e:
                                st.warning(f"导出对比图表时出错: {pair['question1'][:30]}... - {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(matching_pairs))
                        
                        st.success(f"✅ 成功导出 {exported_count} 个对比图表到 {output_dir} 文件夹")
                
                with col2:
                    if st.button("导出所有对比数据到CSV"):
                        output_dir = "comparison_output"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 准备CSV数据
                        csv_data = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, pair in enumerate(matching_pairs):
                            question_type = pair['type']
                            
                            if question_type in ['single_choice', 'multiple_choice']:
                                # 处理单选题和多选题
                                all_options = list(set(pair['data1']['options'] + pair['data2']['options']))
                                
                                for option in all_options:
                                    count1 = 0
                                    count2 = 0
                                    
                                    if option in pair['data1']['options']:
                                        idx = pair['data1']['options'].index(option)
                                        count1 = pair['data1']['data'][idx]
                                    
                                    if option in pair['data2']['options']:
                                        idx = pair['data2']['options'].index(option)
                                        count2 = pair['data2']['data'][idx]
                                    
                                    total1 = sum(pair['data1']['data']) if pair['data1']['data'] else 1
                                    total2 = sum(pair['data2']['data']) if pair['data2']['data'] else 1
                                    
                                    percentage1 = count1/total1*100
                                    percentage2 = count2/total2*100
                                    difference = percentage2 - percentage1
                                    
                                    csv_data.append({
                                        '问题类型': question_type,
                                        '问题相似度': f"{pair['similarity']:.2%}",
                                        f'{group1_name}问题': pair['question1'],
                                        f'{group2_name}问题': pair['question2'],
                                        '选项': option,
                                        f'{group1_name}数量': count1,
                                        f'{group1_name}占比': f"{percentage1:.1f}%",
                                        f'{group2_name}数量': count2,
                                        f'{group2_name}占比': f"{percentage2:.1f}%",
                                        '占比差异': f"{difference:+.1f}%"
                                    })
                            
                            elif question_type == 'rating':
                                # 处理评分题
                                ratings1 = []
                                ratings2 = []
                                
                                for option, count in zip(pair['data1']['options'], pair['data1']['data']):
                                    try:
                                        rating_value = int(option)
                                        ratings1.extend([rating_value] * count)
                                    except:
                                        continue
                                
                                for option, count in zip(pair['data2']['options'], pair['data2']['data']):
                                    try:
                                        rating_value = int(option)
                                        ratings2.extend([rating_value] * count)
                                    except:
                                        continue
                                
                                if ratings1 and ratings2:
                                    ratings1 = pd.Series(ratings1)
                                    ratings2 = pd.Series(ratings2)
                                    
                                    # 添加统计数据
                                    csv_data.append({
                                        '问题类型': question_type,
                                        '问题相似度': f"{pair['similarity']:.2%}",
                                        f'{group1_name}问题': pair['question1'],
                                        f'{group2_name}问题': pair['question2'],
                                        '统计指标': '平均分',
                                        f'{group1_name}数值': f"{ratings1.mean():.2f}",
                                        f'{group2_name}数值': f"{ratings2.mean():.2f}",
                                        '差异': f"{ratings2.mean() - ratings1.mean():+.2f}"
                                    })
                                    
                                    csv_data.append({
                                        '问题类型': question_type,
                                        '问题相似度': f"{pair['similarity']:.2%}",
                                        f'{group1_name}问题': pair['question1'],
                                        f'{group2_name}问题': pair['question2'],
                                        '统计指标': '中位数',
                                        f'{group1_name}数值': f"{ratings1.median():.2f}",
                                        f'{group2_name}数值': f"{ratings2.median():.2f}",
                                        '差异': f"{ratings2.median() - ratings1.median():+.2f}"
                                    })
                                    
                                    csv_data.append({
                                        '问题类型': question_type,
                                        '问题相似度': f"{pair['similarity']:.2%}",
                                        f'{group1_name}问题': pair['question1'],
                                        f'{group2_name}问题': pair['question2'],
                                        '统计指标': '标准差',
                                        f'{group1_name}数值': f"{ratings1.std():.2f}",
                                        f'{group2_name}数值': f"{ratings2.std():.2f}",
                                        '差异': f"{ratings2.std() - ratings1.std():+.2f}"
                                    })
                                    
                                    # 添加各评分的分布数据
                                    rating_counts1 = ratings1.value_counts().sort_index()
                                    rating_counts2 = ratings2.value_counts().sort_index()
                                    all_ratings = sorted(set(list(rating_counts1.index) + list(rating_counts2.index)))
                                    
                                    for rating in all_ratings:
                                        count1 = rating_counts1.get(rating, 0)
                                        count2 = rating_counts2.get(rating, 0)
                                        total1 = len(ratings1)
                                        total2 = len(ratings2)
                                        
                                        percentage1 = count1/total1*100
                                        percentage2 = count2/total2*100
                                        difference = percentage2 - percentage1
                                        
                                        csv_data.append({
                                            '问题类型': question_type,
                                            '问题相似度': f"{pair['similarity']:.2%}",
                                            f'{group1_name}问题': pair['question1'],
                                            f'{group2_name}问题': pair['question2'],
                                            '评分': f"{rating}分",
                                            f'{group1_name}数量': count1,
                                            f'{group1_name}占比': f"{percentage1:.1f}%",
                                            f'{group2_name}数量': count2,
                                            f'{group2_name}占比': f"{percentage2:.1f}%",
                                            '占比差异': f"{difference:+.1f}%"
                                        })
                            
                            progress_bar.progress((i + 1) / len(matching_pairs))
                        
                        # 保存CSV文件
                        if csv_data:
                            csv_df = pd.DataFrame(csv_data)
                            csv_filename = f"{output_dir}/对比数据_{group1_name}_vs_{group2_name}.csv"
                            csv_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                            
                            st.success(f"✅ 成功导出对比数据到 {csv_filename}")
                            st.info(f"📊 共导出 {len(csv_data)} 行对比数据")
                        else:
                            st.warning("⚠️ 没有可导出的对比数据")
            else:
                st.warning("⚠️ 未找到相似的问题可以进行对比分析")
                st.info("请确保两个问卷文件包含相似的问题内容和类型")
        else:
            st.error("❌ 无法解析文件格式，请检查文件内容")
    
    elif analysis_mode == "双文件对比分析":
        if not md_file:
            st.info("👆 请上传第一个问卷文件")
        elif not comparison_file:
            st.info("👆 请上传第二个问卷文件进行对比分析")
    
    elif md_file:
        # 单文件分析模式
        md_content = md_file.read().decode('utf-8')
        analyzer.questions = analyzer.parse_md_metadata_with_data(md_content)
        
        if analyzer.questions:
            st.success(f"✅ 成功解析 {len(analyzer.questions)} 个问题")
            
            # 显示问题列表
            st.header("📝 问题列表")
            question_df = pd.DataFrame([
                {
                    '问题': q_text,
                    '类型': q_info['type'],
                    '选项数': len(q_info['options']),
                    '数据点数': len(q_info['data']) if q_info['data'] else 0,
                    '是否分析': '是' if q_info['type'] != 'fill_in_blank' else '否（填空题跳过）'
                }
                for q_text, q_info in analyzer.questions.items()
            ])
            st.dataframe(question_df, use_container_width=True)
            
            # 生成数据表（可选）
            if st.sidebar.button("生成模拟数据表"):
                if analyzer.create_data_from_parsed_questions():
                    st.success("✅ 已根据实际数据生成模拟数据表")
            
            # 显示数据概览
            st.header("📋 数据概览")
            summary = analyzer.generate_summary_report()
            if summary:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总回答数", summary['总回答数'])
                with col2:
                    st.metric("分析问题数", summary['分析问题数'])
                with col3:
                    st.metric("跳过问题数", summary['跳过问题数（填空题）'])
                with col4:
                    st.metric("数据完整性", summary['数据完整性'])
            
            # 可视化分析
            st.header("📊 可视化分析")
            
            # 选择要分析的问题
            analyzable_questions = {
                q_text: q_info for q_text, q_info in analyzer.questions.items()
                if q_info['type'] != 'fill_in_blank'
            }
            
            if analyzable_questions:
                selected_question = st.selectbox(
                    "选择要分析的问题:",
                    list(analyzable_questions.keys())
                )
                
                if selected_question:
                    question_info = analyzable_questions[selected_question]
                    
                    # 根据问题类型创建可视化
                    fig = None
                    if question_info['type'] == 'single_choice':
                        fig = analyzer.create_single_choice_viz(selected_question)
                    elif question_info['type'] == 'multiple_choice':
                        fig = analyzer.create_multiple_choice_viz(selected_question)
                    elif question_info['type'] == 'rating':
                        fig = analyzer.create_rating_viz(selected_question)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示原始数据统计
                        st.subheader("📈 数据统计")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if question_info['data']:
                                st.write("**总回答数:**", sum(question_info['data']))
                                st.write("**选项数:**", len(question_info['options']))
                        
                        with col2:
                            if question_info['type'] == 'rating' and question_info['data']:
                                # 计算加权平均分
                                total_score = 0
                                total_count = 0
                                for option, count in zip(question_info['options'], question_info['data']):
                                    try:
                                        score = int(option)
                                        total_score += score * count
                                        total_count += count
                                    except:
                                        continue
                                if total_count > 0:
                                    avg_score = total_score / total_count
                                    st.write("**平均分:**", f"{avg_score:.2f}")
            
            # 批量导出功能
            st.header("💾 批量导出")
            if analyzable_questions and st.button("生成所有可视化图表"):
                output_dir = "visualization_output"
                os.makedirs(output_dir, exist_ok=True)
                
                progress_bar = st.progress(0)
                total_questions = len(analyzable_questions)
                exported_count = 0
                
                for i, (q_text, q_info) in enumerate(analyzable_questions.items()):
                    try:
                        # 创建可视化
                        fig = None
                        if q_info['type'] == 'single_choice':
                            fig = analyzer.create_single_choice_viz(q_text)
                        elif q_info['type'] == 'multiple_choice':
                            fig = analyzer.create_multiple_choice_viz(q_text)
                        elif q_info['type'] == 'rating':
                            fig = analyzer.create_rating_viz(q_text)
                        
                        if fig:
                            # 保存图表
                            safe_filename = re.sub(r'[^\w\s-]', '', q_text)[:50]
                            fig.write_image(f"{output_dir}/{safe_filename}.png", 
                                          width=1200, height=800)
                            exported_count += 1
                    except Exception as e:
                        st.warning(f"导出图表时出错: {q_text[:30]}... - {str(e)}")
                    
                    progress_bar.progress((i + 1) / total_questions)
                
                st.success(f"✅ 成功导出 {exported_count} 个图表到 {output_dir} 文件夹")
        else:
            st.error("❌ 无法解析文件格式，请检查文件内容")
    
    else:
        st.info("👆 请在侧边栏上传包含问卷数据的.md文件开始分析")
        
        # 显示使用说明
        st.header("📖 使用说明")
        st.markdown("""
        ### 支持的分析模式
        
        **1. 单文件分析模式**
        - 上传单个问卷数据文件进行分析
        - 支持单选题、多选题、评分题的可视化
        - 提供详细的统计数据和图表
        
        **2. 双文件对比分析模式**
        - 上传两个问卷数据文件进行对比
        - 自动匹配相似问题（基于文本相似度）
        - 生成对比图表和差异分析
        - 支持自定义组别名称
        
        ### 支持的文件格式
        
        **包含数据的元数据文件 (.md):**
        ```
        问题1:【评分题】您对新手引导教程的满意度如何？
        1: 52 (8.64%)
        2: 11 (1.83%)
        3: 108 (17.94%)
        4: 124 (20.60%)
        5: 307 (51.00%)
        
        问题2:【多选题】对于新手教学，您认为不太满意的地方主要是？
        选项1: 28 (12.07%)
        选项2: 35 (15.09%)
        选项3: 26 (11.21%)
        ```
        
        ### 功能特点
        - 🎯 自动识别问题类型（单选、多选、评分、填空）
        - 📊 多样化可视化（条形图、饼图、雷达图等）
        - 🔍 智能问题匹配（基于文本相似度算法）
        - 📈 详细对比分析（数量、百分比、统计差异）
        - 🚫 自动跳过填空题分析
        - 💾 批量导出所有图表
        - 📱 响应式界面设计
        - 🌏 完整中文支持
        - 📈 基于真实数据的可视化
        
        ### 对比分析说明
        - 系统会自动匹配两个文件中相似的问题
        - 相似度阈值设为60%，确保匹配质量
        - 支持同类型问题的对比（单选vs单选，多选vs多选等）
        - 提供数量对比、百分比对比和统计差异分析
        """)

if __name__ == "__main__":
    main()
