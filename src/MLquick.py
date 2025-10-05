import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import zipfile
import shutil
from datetime import datetime
import base64
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import numpy as np


def generate_model_id():
    """生成基于日期时间的模型ID"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_model_files():
    """获取所有可用的模型文件"""
    model_files = []
    models_dir = "../models"
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pkl'):
                model_name = file.replace('.pkl', '')
                model_files.append(model_name)
    return sorted(model_files, reverse=True)  # 最新的在前


def save_model_with_id(model, task_type, model_info=None):
    """保存模型并添加日期时间ID"""
    model_id = generate_model_id()
    model_name = f"{task_type}_model_{model_id}"
    models_dir = "../models"

    # 确保models目录存在
    os.makedirs(models_dir, exist_ok=True)

    # 保存模型
    if task_type == "classification":
        from pycaret.classification import save_model as save_clf_model
        save_clf_model(model, f"{models_dir}/{model_name}")
    elif task_type == "regression":
        from pycaret.regression import save_model as save_reg_model
        save_reg_model(model, f"{models_dir}/{model_name}")
    elif task_type == "clustering":
        from pycaret.clustering import save_model as save_cluster_model
        save_cluster_model(model, f"{models_dir}/{model_name}")

    # 保存模型信息
    info_path = f"{models_dir}/{model_name}_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"模型名称: {model_name}\n")
        f.write(f"任务类型: {task_type}\n")
        f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if model_info:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")

    return model_name


def create_clustering_visualizations(data, cluster_labels, n_clusters):
    """创建聚类可视化图表"""
    visualizations = {}

    # 获取数值型列
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) >= 2:
        # 1. 散点图（前两个主要特征）
        fig1 = px.scatter(
            data,
            x=numeric_columns[0],
            y=numeric_columns[1],
            color=cluster_labels,
            title=f"聚类结果散点图 ({numeric_columns[0]} vs {numeric_columns[1]})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        visualizations['scatter'] = fig1

    if len(numeric_columns) >= 3:
        # 2. 3D散点图（前三个主要特征）
        fig2 = px.scatter_3d(
            data,
            x=numeric_columns[0],
            y=numeric_columns[1],
            z=numeric_columns[2],
            color=cluster_labels,
            title=f"3D聚类结果 ({numeric_columns[0]} vs {numeric_columns[1]} vs {numeric_columns[2]})",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        visualizations['scatter_3d'] = fig2

    # 3. 聚类分布饼图
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    fig3 = px.pie(
        values=cluster_counts.values,
        names=[f'聚类 {i}' for i in cluster_counts.index],
        title='各聚类样本分布'
    )
    visualizations['pie'] = fig3

    # 4. 聚类中心热力图（如果有足够特征）
    if len(numeric_columns) >= 2:
        data_with_clusters = data.copy()
        data_with_clusters['Cluster'] = cluster_labels

        # 计算每个聚类的中心点
        cluster_centers = data_with_clusters.groupby('Cluster')[numeric_columns].mean()

        fig4 = px.imshow(
            cluster_centers.T,
            labels=dict(x="聚类", y="特征", color="均值"),
            title="聚类中心热力图",
            color_continuous_scale='RdYlBu_r'
        )
        visualizations['heatmap'] = fig4

    return visualizations


# K-means聚类任务函数
def clustering_task(data, n_clusters, features=None):
    from pycaret.clustering import setup, create_model, assign_model, pull, plot_model
    from pycaret.clustering import save_model as save_cluster_model

    # 数据预处理：只选择数值型特征
    numeric_data = data.select_dtypes(include=[np.number])

    if features:
        # 如果用户选择了特定特征
        available_features = [f for f in features if f in numeric_data.columns]
        if available_features:
            numeric_data = numeric_data[available_features]

    if numeric_data.empty:
        st.error("❌ 没有找到可用的数值型特征进行聚类分析")
        return None, None, None

    # 设置聚类环境
    setup(data=numeric_data, session_id=123, normalize=True, verbose=False)

    # 创建K-means模型
    with st.spinner("正在训练K-means聚类模型..."):
        kmeans_model = create_model('kmeans', num_clusters=n_clusters)

    st.success("✅ 聚类模型训练完成！")

    # 分配聚类标签
    clustered_data = assign_model(kmeans_model)

    # 创建可视化
    visualizations = create_clustering_visualizations(numeric_data, clustered_data['Cluster'], n_clusters)

    # 保存模型信息
    model_info = {
        "数据集大小": f"{len(data)} 行",
        "原始特征数量": f"{len(data.columns)} 个",
        "数值特征数量": f"{len(numeric_data.columns)} 个",
        "聚类数量": n_clusters,
        "聚类算法": "K-means",
        "使用的特征": ", ".join(numeric_data.columns.tolist())
    }

    # 计算聚类统计信息
    cluster_stats = clustered_data.groupby('Cluster').agg({
        col: ['mean', 'std', 'count'] for col in numeric_data.columns
    }).round(3)

    model_info["聚类统计"] = f"已生成各聚类的统计信息"

    # 使用新的保存函数
    model_name = save_model_with_id(kmeans_model, "clustering", model_info)
    st.session_state.current_model_name = model_name

    # 保存聚类结果
    clustered_data.to_csv(f"../models/{model_name}_results.csv", index=False)

    return kmeans_model, clustered_data, model_name, visualizations, cluster_stats


# 分类任务函数
def classification_task(data, target_variable, train_size):
    from pycaret.classification import setup, compare_models, save_model, pull, plot_model, predict_model
    from pycaret.classification import save_model as save_clf_model

    setup(data=data, target=target_variable, session_id=123, normalize=True, train_size=train_size)
    best_model = compare_models()
    st.success("✅ 模型训练完成！")

    # 保存模型信息
    model_comparison = pull()
    best_model_name = str(best_model)
    accuracy = model_comparison.loc['Accuracy', best_model_name] if 'Accuracy' in model_comparison.index else 'N/A'

    model_info = {
        "数据集大小": f"{len(data)} 行",
        "特征数量": f"{len(data.columns) - 1} 个",
        "目标变量": target_variable,
        "训练集比例": f"{train_size:.1%}",
        "最佳模型": best_model_name,
        "准确率": f"{accuracy:.4f}" if accuracy != 'N/A' else 'N/A'
    }

    # 使用新的保存函数
    model_name = save_model_with_id(best_model, "classification", model_info)
    st.session_state.current_model_name = model_name

    return best_model, model_comparison, model_name


# 回归任务函数
def regression_task(data, target_variable, train_size):
    from pycaret.regression import setup, compare_models, save_model, pull, predict_model
    from pycaret.regression import save_model as save_reg_model

    setup(data=data, target=target_variable, train_size=train_size)
    best_model = compare_models()
    st.success("✅ 模型训练完成！")

    # 保存模型信息
    model_comparison = pull()
    best_model_name = str(best_model)
    r2 = model_comparison.loc['R2', best_model_name] if 'R2' in model_comparison.index else 'N/A'
    rmse = model_comparison.loc['RMSE', best_model_name] if 'RMSE' in model_comparison.index else 'N/A'

    model_info = {
        "数据集大小": f"{len(data)} 行",
        "特征数量": f"{len(data.columns) - 1} 个",
        "目标变量": target_variable,
        "训练集比例": f"{train_size:.1%}",
        "最佳模型": best_model_name,
        "R² 分数": f"{r2:.4f}" if r2 != 'N/A' else 'N/A',
        "RMSE": f"{rmse:.4f}" if rmse != 'N/A' else 'N/A'
    }

    # 使用新的保存函数
    model_name = save_model_with_id(best_model, "regression", model_info)
    st.session_state.current_model_name = model_name

    return best_model, model_comparison, model_name


# 预测函数
def prediction(model_path, prediction_file):
    try:
        models_dir = "../models"
        full_model_path = f"{models_dir}/{model_path}"

        if os.path.exists(f'{full_model_path}.pkl'):
            if 'classification' in model_path:
                from pycaret.classification import load_model, predict_model
            elif 'regression' in model_path:
                from pycaret.regression import load_model, predict_model
            elif 'clustering' in model_path:
                from pycaret.clustering import load_model, assign_model
                # 聚类任务的特殊处理
                loaded_model = load_model(full_model_path)
                st.success("✅ 聚类模型已成功载入")

                # 读取待预测数据
                if prediction_file.name.endswith('.csv'):
                    prediction_data = pd.read_csv(prediction_file, encoding='utf-8-sig')
                elif prediction_file.name.endswith('.xlsx'):
                    prediction_data = pd.read_excel(prediction_file, engine='openpyxl')

                # 只保留数值型特征
                numeric_prediction_data = prediction_data.select_dtypes(include=[np.number])

                # 进行聚类预测
                clustered_prediction = assign_model(loaded_model, data=numeric_prediction_data)
                st.success("✅ 聚类预测完成！")
                st.write("聚类结果：")
                st.dataframe(clustered_prediction)

                # 提供下载功能
                csv = clustered_prediction.to_csv(index=False)
                st.download_button(
                    label="下载聚类结果 (CSV)",
                    data=csv,
                    file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                return

            # 分类和回归任务的通用处理
            loaded_model = load_model(full_model_path)
            st.success("✅ 模型已成功载入")

            # 读取待预测数据
            if prediction_file.name.endswith('.csv'):
                prediction_data = pd.read_csv(prediction_file, encoding='utf-8-sig')
            elif prediction_file.name.endswith('.xlsx'):
                prediction_data = pd.read_excel(prediction_file, engine='openpyxl')

            predictions = predict_model(loaded_model, data=prediction_data)
            st.success("✅ 预测完成！")
            st.write("预测结果：")
            st.dataframe(predictions)

            # 提供下载功能
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="下载预测结果 (CSV)",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        else:
            st.error("❌ 未找到相应的模型文件，请先训练模型或选择正确的模型")
    except Exception as e:
        st.error(f"❌ 预测过程中出现错误: {str(e)}")


def export_model(model_name):
    """导出模型文件"""
    try:
        models_dir = "../models"
        model_path = f"{models_dir}/{model_name}"

        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 复制模型文件
            shutil.copy(f"{model_path}.pkl", temp_dir)

            # 复制模型信息文件（如果存在）
            info_file = f"{model_path}_info.txt"
            if os.path.exists(info_file):
                shutil.copy(info_file, temp_dir)

            # 复制聚类结果文件（如果存在）
            results_file = f"{model_path}_results.csv"
            if os.path.exists(results_file):
                shutil.copy(results_file, temp_dir)

            # 创建zip文件
            zip_path = f"{temp_dir}/{model_name}.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(f"{temp_dir}/{model_name}.pkl", f"{model_name}.pkl")
                if os.path.exists(info_file):
                    zipf.write(f"{temp_dir}/{model_name}_info.txt", f"{model_name}_info.txt")
                if os.path.exists(results_file):
                    zipf.write(f"{temp_dir}/{model_name}_results.csv", f"{model_name}_results.csv")

            # 读取zip文件并返回
            with open(zip_path, 'rb') as f:
                zip_data = f.read()

            return zip_data
    except Exception as e:
        st.error(f"导出模型时出现错误: {str(e)}")
        return None


def import_model(uploaded_file):
    """导入模型文件"""
    try:
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)

        # 保存上传的文件
        temp_path = f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 解压文件
        with zipfile.ZipFile(temp_path, 'r') as zipf:
            zipf.extractall(models_dir)

        # 删除临时文件
        os.remove(temp_path)

        st.success("✅ 模型导入成功！")
        return True
    except Exception as e:
        st.error(f"导入模型时出现错误: {str(e)}")
        return False


def show_model_info(model_name):
    """显示模型信息"""
    try:
        models_dir = "../models"
        info_file = f"{models_dir}/{model_name}_info.txt"

        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = f.read()
            st.info(f"📋 **模型信息**\n\n```\n{info}\n```")
        else:
            st.info("📋 模型信息文件不存在")
    except Exception as e:
        st.warning(f"读取模型信息时出现错误: {str(e)}")


# 定义主函数
def main():
    st.title("MLquick - 机器学习零代码应用平台")

    # 侧边栏 - 模型管理
    st.sidebar.markdown("## 🔧 模型管理")

    # 显示当前模型
    if 'current_model_name' in st.session_state and st.session_state.current_model_name:
        st.sidebar.success(f"当前模型: {st.session_state.current_model_name}")
        show_model_info(st.session_state.current_model_name)

    # 模型导出
    if 'current_model_name' in st.session_state and st.session_state.current_model_name:
        if st.sidebar.button("📤 导出当前模型"):
            zip_data = export_model(st.session_state.current_model_name)
            if zip_data:
                st.sidebar.download_button(
                    label=f"下载 {st.session_state.current_model_name}",
                    data=zip_data,
                    file_name=f"{st.session_state.current_model_name}.zip",
                    mime="application/zip"
                )

    # 模型导入
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 导入模型")
    uploaded_model = st.sidebar.file_uploader("上传模型文件 (.zip)", type=["zip"])
    if uploaded_model is not None:
        if st.sidebar.button("导入模型"):
            if import_model(uploaded_model):
                st.rerun()

    # 可用模型列表
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 可用模型")
    model_files = get_model_files()
    if model_files:
        selected_model = st.sidebar.selectbox("选择模型", model_files)
        if st.sidebar.button("加载选中模型"):
            st.session_state.current_model_name = selected_model
            st.rerun()
    else:
        st.sidebar.info("暂无可用模型")

    # 主界面
    st.markdown("---")

    # 上传数据
    uploaded_file = st.file_uploader("📁 上传数据集 (CSV 或 Excel格式)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # 判断文件类型并读取数据
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')

        data = pd.DataFrame(data)
        st.markdown("### 📊 数据预览")
        st.write(f"数据形状: {data.shape[0]} 行 × {data.shape[1]} 列")
        st.dataframe(data.head(10))

        # 数据基本信息
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        st.info(f"📈 **数据统计**: 数值型特征 {len(numeric_columns)} 个，总特征 {len(data.columns)} 个")

        # 选择任务类型
        st.markdown("### ⚙️ 模型配置")
        task_type = st.selectbox("选择任务类型", ["分类", "回归", "聚类"])

        if task_type == "聚类":
            # 聚类任务特殊配置
            st.markdown("#### 🎯 聚类配置")

            # 聚类数量
            n_clusters = st.number_input(
                "聚类数量 (K值)",
                min_value=2,
                max_value=min(20, len(data)),
                value=3,
                step=1,
                help="K-means聚类的类别数量"
            )

            # 特征选择
            if numeric_columns:
                selected_features = st.multiselect(
                    "选择用于聚类的特征 (留空则使用所有数值特征)",
                    numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))],  # 默认选择前5个特征
                    help="选择用于聚类分析的数值型特征"
                )
            else:
                st.warning("⚠️ 数据中没有数值型特征，无法进行聚类分析")
                selected_features = []

        else:
            # 分类和回归任务的配置
            target_variable = st.selectbox("选择目标变量", data.columns)
            train_size = st.number_input("输入训练集比例（0 - 1之间）", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

        # 初始化会话状态
        if 'best_model' not in st.session_state:
            st.session_state.best_model = None
        if 'model_comparison' not in st.session_state:
            st.session_state.model_comparison = None
        if 'clustered_data' not in st.session_state:
            st.session_state.clustered_data = None
        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = None

        # 训练模型
        if st.button("🚀 开始训练模型", type="primary"):
            with st.spinner("正在训练模型，请稍候..."):
                if task_type == "聚类":
                    if not numeric_columns:
                        st.error("❌ 没有数值型特征可用于聚类分析")
                    else:
                        model, clustered_data, model_name, visualizations, cluster_stats = clustering_task(
                            data, n_clusters, selected_features)
                        if model is not None:
                            st.session_state.best_model = model
                            st.session_state.clustered_data = clustered_data
                            st.session_state.visualizations = visualizations
                            st.session_state.cluster_stats = cluster_stats

                elif task_type == "分类":
                    st.session_state.best_model, st.session_state.model_comparison, model_name = classification_task(
                        data, target_variable, train_size)
                else:  # 回归
                    st.session_state.best_model, st.session_state.model_comparison, model_name = regression_task(
                        data, target_variable, train_size)

        # 显示结果
        if task_type == "聚类" and st.session_state.clustered_data is not None:
            st.markdown("### 📈 聚类分析结果")

            # 显示聚类统计信息
            if 'cluster_stats' in st.session_state:
                st.markdown("#### 📊 聚类统计信息")
                st.dataframe(st.session_state.cluster_stats)

            # 显示可视化
            if st.session_state.visualizations:
                st.markdown("#### 🎨 聚类可视化")

                # 散点图
                if 'scatter' in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations['scatter'], use_container_width=True)

                # 3D散点图
                if 'scatter_3d' in st.session_state.visualizations:
                    st.plotly_chart(st.session_state.visualizations['scatter_3d'], use_container_width=True)

                # 饼图
                if 'pie' in st.session_state.visualizations:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(st.session_state.visualizations['pie'], use_container_width=True)

                # 热力图
                if 'heatmap' in st.session_state.visualizations:
                    with col2:
                        st.plotly_chart(st.session_state.visualizations['heatmap'], use_container_width=True)

            # 下载聚类结果
            csv = st.session_state.clustered_data.to_csv(index=False)
            st.download_button(
                label="📥 下载聚类结果 (CSV)",
                data=csv,
                file_name=f"clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        elif task_type != "聚类" and st.session_state.model_comparison is not None:
            st.markdown("### 📈 模型性能对比")
            st.dataframe(st.session_state.model_comparison)

        # 预测功能
        st.markdown("---")
        st.markdown("### 🔮 模型预测")

        # 选择预测方式
        prediction_mode = st.radio("选择预测方式", ["使用当前训练的模型", "选择已有模型"])

        if prediction_mode == "使用当前训练的模型":
            if 'current_model_name' in st.session_state and st.session_state.current_model_name:
                st.info(f"使用模型: {st.session_state.current_model_name}")
                prediction_file = st.file_uploader("📁 上传待预测数据", type=["csv", "xlsx"], key="pred_current")
                if prediction_file is not None:
                    prediction(st.session_state.current_model_name, prediction_file)
            else:
                st.warning("请先训练模型")

        else:  # 选择已有模型
            model_files = get_model_files()
            if model_files:
                selected_model = st.selectbox("选择模型", model_files, key="pred_model_select")
                st.info(f"使用模型: {selected_model}")
                prediction_file = st.file_uploader("📁 上传待预测数据", type=["csv", "xlsx"], key="pred_existing")
                if prediction_file is not None:
                    prediction(selected_model, prediction_file)
            else:
                st.warning("暂无可用模型，请先训练模型或导入模型")

    # 页脚
    st.markdown("---")
    st.markdown("### 💡 使用提示")
    st.markdown("""
    - **分类任务**: 需要选择目标变量，用于预测类别
    - **回归任务**: 需要选择目标变量，用于预测数值
    - **聚类任务**: 自动发现数据中的群组，无需目标变量
    - 支持的文件格式: CSV (.csv), Excel (.xlsx)
    - 模型会自动保存，不会覆盖之前的模型
    - 可以通过侧边栏导入/导出模型
    - 预测结果可以下载为CSV文件
    - 建议训练集比例设置在0.6-0.8之间（仅对分类和回归任务）
    """)


if __name__ == "__main__":
    main()