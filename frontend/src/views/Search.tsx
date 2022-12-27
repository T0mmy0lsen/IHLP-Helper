import React from "react";

import {
    Typography,
    ListHeader,
    Formula,
    Section,
    Button,
    Action,
    Space,
    Title,
    List,
    Post,
    Get, Main, ListItem,
    Autocomplete, Conditions, ConditionsItem, Item, Search, Input
} from 'react-antd-admin-panel/dist';

import {SectionComponent} from 'react-antd-admin-panel/dist';

import {Col, DatePicker, message, Row} from "antd";
import {ArrowRightOutlined} from "@ant-design/icons";
import { useState } from "react";

export default class Home extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            time: false,
            text: '',
            id: false,
        }
    }

    build() {

        const getSearch = () => {
            let input = new Input()
                .onChange((v) => this.setState({text: v}))
                .onPressEnter(() => {
                    let get = new Get()
                        .target(() => ({
                            target: `/request`,
                            params: {
                                text: this.state.text,
                                time: this.state.time ? this.state.time.format('YYYY-MM-DD HH:MM:ss') : false
                            }
                        }))
                        .onComplete((v: any) => {
                            let data = v.data.data
                            if (data.length > 0) {
                                condition.checkCondition({
                                    data: data,
                                    loading: false
                                })
                            }
                            input.tsxSetDisabled(false)
                        });
                    get.get()
                    input.tsxSetDisabled(true)
                })

            return input
        }

        const getList = (args) => new List()
            .unique((v) => v.id)
            .default({ dataSource: args })
            .footer(false)
            .headerCreate(false)
            .headerPrepend(new ListHeader().key('id').title('Request').sortable())
            .headerPrepend(new ListHeader().key('subject').title('Subject').searchable())
            .expandable(() => true)
            .expandableExpandAll()
            .expandableSection((item: ListItem) => {
                let data = item._object.predict.data.list
                let list: any[] = []
                Object.keys(data).forEach((key) => {
                    list.push({
                        user: key,
                        weight: data[key].predictions_sum,
                        estimate: data[key].predictions_index[0],
                    })
                })
                list = list.sort((a, b) => b.weight - a.weight)
                list = list.slice(0, 10)
                let section = new Section().style({ padding: '8px' })
                section.add(new List()
                    .unique((v) => v.user)
                    .default({ dataSource: list })
                    .footer(false)
                    .headerCreate(false)
                    .headerPrepend(new ListHeader().key('user').title('User').searchable())
                    .headerPrepend(new ListHeader().key('weight').title('Weight').sortable())
                    .headerPrepend(new ListHeader().key('estimate').title('Estimate').searchable())
                )
                return section;
            })

        const main: Main = this.props.main;
        const search = getSearch();
        const section = new Section();

        const Time = () => {

            const onOk = (value) => {
                this.setState({ time: value })
            };

            return (
                <DatePicker showTime onOk={onOk} />
            )
        }

        let condition = new Conditions()
            .default(() => ({ value: undefined, loading: false }))
            .add(new ConditionsItem()
                .condition((v: any) => !!v.data)
                .content((next, callback, main, args) => {
                    next(new Section()
                        .add(new Space().top(24))
                        .add(getList(args.data))
                    )
                })
            )
            .add(new ConditionsItem()
                .condition((v: any) => !v.data)
                .content((next, callback, main, args) => next(new Section()))
            );

        section.style({ padding: '24px 36px' });
        section.add(new Title().label('Search for a Request').level(1));
        section.add(new Section().component(Time, false))
        section.add(new Space().top(24));
        section.add(search);
        section.add(condition);

        this.setState({ section: section });
    }

    render() {
        return (
            <>{!!this.state.section &&
            <SectionComponent key={this.state.id} main={this.props.main} section={this.state.section}/>}</>
        );
    }

    componentDidMount() {
        this.build()
    }
}