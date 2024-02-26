import pandas as pd
import os
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def preprocess_tasks(tasks: pd.DataFrame) -> pd.DataFrame:
    # convert ref column to task_id
    tasks["task_id"] = tasks["ref"]
    tasks = tasks.drop(columns=["ref"])
    tasks["userstory_id"] = tasks["user_story"].astype(int)
    tasks = tasks.drop(columns=["user_story"])
    return tasks


def preprocess_userstories(userstories: pd.DataFrame) -> pd.DataFrame:
    # convert ref column to userstory_id
    userstories["userstory_id"] = userstories["ref"].astype(int)
    userstories = userstories.drop(columns=["ref"])
    return userstories


def preprocess_epics(epics: pd.DataFrame) -> pd.DataFrame:
    epics["epic_id"] = epics["ref"].astype(int)
    epics = epics.drop(columns=["ref"])
    epics["userstory_id"] = epics["related_user_stories"].str.findall(r"#(\d+)")
    epics = epics.drop(columns=["related_user_stories"])

    epics = epics.explode("userstory_id")
    epics["userstory_id"] = epics["userstory_id"].astype(int)
    epics["epic_name"] = epics["subject"]
    epics = epics.drop(columns=["subject"])
    return epics


def merge_data(
    tasks: pd.DataFrame, userstories: pd.DataFrame, epics: pd.DataFrame
) -> pd.DataFrame:

    merge_df = pd.merge(
        tasks,
        userstories,
        on="userstory_id",
        how="left",
        suffixes=("_task", "_userstory"),
        validate="m:1",
    )
    merge_df = pd.merge(
        merge_df,
        epics,
        on="userstory_id",
        how="left",
        suffixes=(
            "",
            "_epic",
        ),
        validate="m:1",
    )

    return merge_df


def validate_start_date_end_date(date: pd.Timestamp, holiday_list: List[pd.Timestamp]):
    # if start_date is weekend, then change to next Monday
    if date.weekday() in [5, 6]:
        date = date + pd.Timedelta(days=(7 - date.weekday()))
    while date in holiday_list:
        date = date + pd.Timedelta(days=1)
    return date


def get_holiday_list():
    # from outlook email
    holiday_list = [
        "2024/01/01",
        "2024/02/12",
        "2024/02/26",
        "2024/04/08",
        "2024/04/15",
        "2024/04/16",
        "2024/05/01",
        "2024/05/06",
        "2024/05/22",
        "2024/06/03",
        "2024/07/22",
        "2024/07/29",
        "2024/08/12",
        "2024/10/14",
        "2024/10/23",
        "2024/12/05",
        "2024/12/10",
        "2024/12/31",
    ]
    holiday_list = [pd.to_datetime(date) for date in holiday_list]
    return holiday_list


def get_work_day(start_date, end_date, holiday_list=[]):
    # count when it is not weekend
    date_range = pd.date_range(start=start_date, end=end_date)
    work_day = 0
    for date in date_range:
        if date in holiday_list:
            continue
        if date.weekday() in [5, 6]:
            continue
        work_day += 1
    if work_day == 0:
        work_day = 1
    return work_day


def get_task_daily_df(merge_data_df: pd.DataFrame) -> pd.DataFrame:
    task_daily_list = []

    holiday_list = get_holiday_list()
    for _, row in merge_data_df.iterrows():
        start_date = pd.to_datetime(row["start date"])
        end_date = pd.to_datetime(row["end date"])
        start_date = validate_start_date_end_date(start_date, holiday_list)
        end_date = validate_start_date_end_date(end_date, holiday_list)
        if end_date < start_date:
            end_date = start_date
        assignee = row["assigned_to_task"]
        epic_name = row["epic_name"]
        task_id = row["task_id"]
        actual_time = row["actual time"]
        total_date = get_work_day(start_date, end_date, holiday_list)
        hour = actual_time / total_date
        date_range = pd.date_range(start=start_date, end=end_date)
        for date in date_range:
            if date.weekday() in [5, 6]:
                continue
            if date in holiday_list:
                continue
            task_daily_list.append(
                {
                    "date": date,
                    "assignee": assignee,
                    "epic_name": epic_name,
                    "task_id": task_id,
                    "hour": hour,
                }
            )
    task_daily_df = pd.DataFrame(task_daily_list)
    return task_daily_df


def get_assignee_hour_per_day_df(task_daily_df: pd.DataFrame) -> pd.DataFrame:
    assignee_hour_per_day_df = (
        task_daily_df.groupby(["date", "assignee"]).agg({"hour": "sum"}).reset_index()
    )
    # pivot table
    assignee_hour_per_day_df = assignee_hour_per_day_df.pivot(
        index="date", columns="assignee", values="hour"
    )
    # fill na with 0
    assignee_hour_per_day_df = assignee_hour_per_day_df.fillna(0)
    min_date = assignee_hour_per_day_df.index.min()
    max_date = assignee_hour_per_day_df.index.max()
    # fill missing date
    date_range = pd.date_range(start=min_date, end=max_date)
    assignee_hour_per_day_df = assignee_hour_per_day_df.reindex(
        date_range, fill_value=0
    )
    # change column name first column to 'date'
    assignee_hour_per_day_df = assignee_hour_per_day_df.reset_index()
    assignee_hour_per_day_df = assignee_hour_per_day_df.rename(
        columns={"index": "date"}
    )
    return assignee_hour_per_day_df


def get_project_hour_per_day_df(task_daily_df: pd.DataFrame) -> pd.DataFrame:
    project_hour_per_day_df = (
        task_daily_df.groupby(["date", "epic_name"]).agg({"hour": "sum"}).reset_index()
    )
    project_hour_per_day_df = project_hour_per_day_df.pivot(
        index="date", columns="epic_name", values="hour"
    )
    project_hour_per_day_df = project_hour_per_day_df.fillna(0)
    min_date = project_hour_per_day_df.index.min()
    max_date = project_hour_per_day_df.index.max()
    date_range = pd.date_range(start=min_date, end=max_date)
    project_hour_per_day_df = project_hour_per_day_df.reindex(date_range, fill_value=0)
    project_hour_per_day_df = project_hour_per_day_df.reset_index()
    project_hour_per_day_df = project_hour_per_day_df.rename(columns={"index": "date"})
    return project_hour_per_day_df


def creat_fully_report_graph(
    num_date_dict: Dict[str, int],
    pivot_hour_per_day_df: pd.DataFrame,
    holiday_list: List[pd.Timestamp] = get_holiday_list(),
    fig_title: str = "Fully Report",
    is_display_work_hour: bool = True,
) -> go.Figure:
    num_date_list = list(num_date_dict.values())
    column_list = pivot_hour_per_day_df.columns.tolist()
    # remove date
    column_list.remove("date")
    fig = make_subplots(
        rows=len(num_date_list),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
    )
    showlegend = True
    # create color dict  from  column_list
    color_line_dict = {
        column: px.colors.qualitative.Plotly[i] for i, column in enumerate(column_list)
    }
    for plot_ind, (plot_name, num_date) in enumerate(num_date_dict.items()):
        if plot_ind != 0:
            showlegend = False
        report_plot = pivot_hour_per_day_df.tail(num_date)
        report_plot = report_plot.reset_index()

        work_hour_list = [8] * len(report_plot)
        date_list = report_plot["date"].tolist()
        for date in date_list:
            if date in holiday_list:
                work_hour_list[date_list.index(date)] = 0
            if date.weekday() in [5, 6]:
                work_hour_list[date_list.index(date)] = 0

        for column in column_list:
            fig.add_trace(
                go.Scatter(
                    x=report_plot["date"],
                    y=report_plot[column],
                    mode="lines+markers",
                    name=column,
                    showlegend=showlegend,
                    legendgroup=column,
                    line=dict(color=color_line_dict[column]),
                ),
                row=plot_ind + 1,
                col=1,
            )
        if is_display_work_hour:
            fig.add_trace(
                go.Scatter(
                    x=report_plot["date"],
                    y=work_hour_list,
                    mode="lines",
                    name="8 hour",
                    line=dict(color="green", dash="dash"),
                    showlegend=False,
                ),
                row=plot_ind + 1,
                col=1,
            )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=report_plot["date"].tolist(),
                ticktext=report_plot["date"].dt.strftime("%d-%b").tolist(),
                tickangle=45,
            ),
        )
        fig.update_xaxes(
            **dict(
                tickmode="array",
                tickvals=report_plot["date"].tolist(),
                ticktext=report_plot["date"].dt.strftime("%d-%b").tolist(),
                tickangle=45,
            ),
            row=plot_ind + 1,
            col=1,
            title_text=plot_name,
        )

        fig.update_yaxes(title_text="Hour", row=plot_ind + 1, col=1)
        fig.update_layout(
            title=fig_title,
            title_x=0.5,
            title_font_size=30,
        )

    return fig


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tasks_path = f"{dir_path}/data/tasks.csv"
    userstories_path = f"{dir_path}/data/userstories.csv"
    epics_path = f"{dir_path}/data/epics.csv"
    merge_data_df_path = f"{dir_path}/result/merge_data.xlsx"
    project_hour_per_day_report_graph_path = (
        f"{dir_path}/result/project_hour_per_day_report_graph.html"
    )
    assignee_hour_per_day_report_graph_path = (
        f"{dir_path}/result/assignee_hour_per_day_report_graph.html"
    )
    tasks = pd.read_csv(tasks_path)
    userstories = pd.read_csv(userstories_path)
    epics = pd.read_csv(epics_path)

    # preprocess data
    tasks = preprocess_tasks(tasks)
    userstories = preprocess_userstories(userstories)
    epics = preprocess_epics(epics)

    merge_data_df = merge_data(tasks, userstories, epics)
    merge_data_df.to_excel(merge_data_df_path, index=False)
    with pd.ExcelWriter("merge_data.xlsx") as writer:
        merge_data_df.to_excel(writer, sheet_name="merge_data", index=False)

    task_daily_df = get_task_daily_df(merge_data_df)
    assignee_hour_per_day_df = get_assignee_hour_per_day_df(task_daily_df)
    project_hour_per_day_df = get_project_hour_per_day_df(task_daily_df)
    with pd.ExcelWriter("task_daily_df.xlsx") as writer:
        merge_data_df.to_excel(writer, sheet_name="merge_data_df")
        task_daily_df.to_excel(writer, sheet_name="task_daily_df")
        assignee_hour_per_day_df.to_excel(writer, sheet_name="assignee_hour_per_day_df")
        project_hour_per_day_df.to_excel(writer, sheet_name="project_hour_per_day_df")
    num_date_dict = {
        "Bi-Weekly": 14,
        "Monthly": 30,
        "Quarterly": 90,
    }

    assignee_hour_per_day_report_graph = creat_fully_report_graph(
        num_date_dict,
        assignee_hour_per_day_df,
        fig_title="Assignee Hour Per Day Report",
    )
    assignee_hour_per_day_report_graph.write_html(
        assignee_hour_per_day_report_graph_path
    )

    project_hour_per_day_report_graph = creat_fully_report_graph(
        num_date_dict,
        project_hour_per_day_df,
        fig_title="Project Hour Per Day Report",
        is_display_work_hour=False,
    )

    project_hour_per_day_report_graph.write_html(
        project_hour_per_day_report_graph_path,
    )

    print("complete")


if __name__ == "__main__":
    main()
