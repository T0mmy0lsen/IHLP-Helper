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
    Autocomplete, Conditions, ConditionsItem, Item
} from 'react-antd-admin-panel/dist';

import {SectionComponent} from 'react-antd-admin-panel/dist';

import {Col, message, Row} from "antd";
import {ArrowRightOutlined} from "@ant-design/icons";
import { useState } from "react";

export default class Home extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            id: false,
        }
    }

    // DONE - TODO: Complete visuals
    // DONE - TODO: Choose team or single person
    // TODO: Only predict for users that are in teams. Only schedule for users that are in LabelEncoder.
    // TODO: Make reset + boot
    // TODO: Setup with SVM model

    build() {

        const main: Main = this.props.main;
        const search = new Autocomplete();
        const section = new Section();

        const Label = (props: any) => {
            let style = {
                fontSize: 11,
                textAlign: 'left' as const,
                width: '100%'
            }
            return (
                <Row style={style}>
                    <Col span={8} style={{ fontWeight: 700 }}>{ props.args.prob }%</Col>
                    <Col>{props.args.key}</Col>
                </Row>
            )
        };

        const Expand = (props: any) => {
            console.log(props.args)
            let [data, setData] = useState(props.args)
            let style = {
                background: '#0092ff',
                height: 24,
                padding: '0px',
                paddingLeft: '12px',
                margin: 2,
                fontSize: 12,
            }
            let styleBtn = {
                background: 'white',
                height: 24,
                padding: '0px',
                paddingLeft: '12px',
                margin: 2,
                marginRight: 2,
                fontSize: 12,
            }
            return (
                <>
                    { Object.keys(data.schedule).map((key, i) => {
                        let section = new Section()
                        section.add(new Button()
                            .block()
                            .small()
                            .component(Label, { key: key, prob: data.schedule[key]['prob'] })
                            .action(new Action()
                                .callback(() => {
                                    new Get().target(() => ({
                                        target: 'predict',
                                        method: 'post',
                                        params: {
                                            id: search._defaultObject.object.id,
                                            user: key,
                                            time: data['time'],
                                            keep: true
                                        }
                                    }))
                                        .onComplete(() => {
                                            message.success("The choice has been submitted.")
                                            condition.checkCondition({ value: false, loading: false })
                                        })
                                        .get()
                                })
                            )
                        )
                        return <Row key={i}>
                            <Col span={2} style={styleBtn}>
                                <SectionComponent key={i} { ... props } model={section}/>
                            </Col>
                            {  data.schedule[key]['list'].map((e, j) => {
                                let tmpStyle = { ...style, background: (e.current ? style.background : 'green') }
                                return <Col key={j} span={data.schedule[key]['list'][j]['time']} style={tmpStyle}></Col>
                            })}
                        </Row>
                    })}
                </>
            )
        }

        let condition = new Conditions()
            .default(() => ({ value: undefined, loading: false }))
            .add(new ConditionsItem()
                .condition((v: any) => {
                    return !!v.value
                })
                .content((next, callback, main, args) => {
                    next(new Section()
                        .add(new Space().top(24))
                        // List element
                        .add(new List()
                            .unique((v) => v.id)
                            .fetch(() => new Get()
                                .target(() => ({
                                    target: `/predict`,
                                    params: {
                                        id: args.value
                                    }
                                })))
                            .footer(false)
                            .headerCreate(false)
                            .headerPrepend(new ListHeader().key('time').title('Team').searchable())
                            .headerPrepend(new ListHeader().key('team').title('Team Sum').sortable())
                            .headerPrepend(new ListHeader().key('team_prob').title('Team Score').sortable())
                            .headerPrepend(new ListHeader().key('team_time').title('Team Time').sortable())
                            .actions(new Action()
                                .icon(ArrowRightOutlined)
                                .callback((args, _) => {
                                    new Get().target(() => ({
                                        target: 'predict',
                                        method: 'post',
                                        params: {
                                            id: search._defaultObject.object.id,
                                            user: args.record.user,
                                            time: args.record.time,
                                            keep: false
                                        }
                                    }))
                                        .onComplete(() => {
                                            message.success("The choice has been submitted.")
                                            condition.checkCondition({ value: false, loading: false })
                                        })
                                        .get()
                                })
                            )
                            .expandable(() => true)
                            .expandableExpandAll()
                            .expandableSection((item: ListItem) => {
                                let section = new Section().style({ padding: '8px' }).add(new Section()
                                    .component(Expand, item)
                                )
                                return section;
                            })
                        )
                    )
                })
            )
            .add(new ConditionsItem()
                .condition((v: any) => {
                    return !v.value
                })
                .content((next, callback, main, args) => next(new Section()))
            );

        search.key('requestId')
            .label('Find a request')
            .clearable(false)
            .get((value) => new Get()
                .target(() => ({
                    target: `/request`,
                    params: {
                        text: value
                    }
                }))
                .alter((v: any) => v
                    .map((r: any) => new Item(r.id).id(r.id).title(r.subject).text(r.description).object(r))
                )
            )
            .onChange((v: any) => {
                if (v.value) {
                    condition.checkCondition({ value: v.object.id, loading: false })
                }
            });

        section.style({ padding: '24px 36px' });
        section.add(new Title().label('Search for a Request').level(1));
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