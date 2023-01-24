import React from "react";

import {
    Typography,
    ListHeader,
    Formula,
    Section,
    Action,
    Space,
    Title,
    List,
    Post,
    Get, Main, ListItem,
    Autocomplete, Conditions, ConditionsItem, Item, Search, Input, Switch
} from 'react-antd-admin-panel/dist';

import {SectionComponent} from 'react-antd-admin-panel/dist';

import {message, Input as InputFromAntd, Select as SelectFromAntd, Col, Row, Badge, Avatar, Card, Descriptions, Button, Modal} from "antd";
import {ArrowRightOutlined, CheckCircleOutlined, HeartOutlined, MehOutlined, MessageOutlined} from "@ant-design/icons";
import { useState } from "react";

const workload_upper_limit = 2000;
const limit_high = 5.8;
const limit_low = 3.9;

export default class Home extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            limit: 10,
            maxId: 0,
            text: '',
            id: false,
            workload: false,
            isLoading: false,
        }
    }

    build() {

        const callRate = (o, msg, type, onComplete, onError) => {
            new Get()
                .target(() => {
                    return ({
                        target: `/message`,
                        params: {
                            id: o.id,
                            type: type,
                            message: msg,
                        }
                    })
                })
                .onComplete((v: any) => {
                    onComplete()
                    let data = v
                    message.success('Thanks you for the feedback!')
                })
                .onError(() => {
                    onError()
                    message.error('The action failed.')
                })
                .get()
        }

        const ModalMessage = (o) => {

            const [isModalOpen, setIsModalOpen] = useState(false);
            const [value, setValue] = useState('');
            const [loading, setLoading] = useState(false);

            const showModal = () => {
                setIsModalOpen(true);
            };

            const handleOk = (e) => {
                e.stopPropagation();
                e.preventDefault();
                setLoading(true)
                callRate(o.o, value, o.type,
                    () => {
                        setLoading(false)
                        setIsModalOpen(false);
                        message.success('Thanks!')
                    },
                    () => {
                        setLoading(false)
                        message.error('The action failed.')
                    },
                )
            };

            const handleCancel = (e) => {
                e.stopPropagation();
                e.preventDefault();
                setIsModalOpen(false);
            };

            return (
                <>
                    <Button
                        type="ghost"
                        icon={<MessageOutlined/>}
                        onClick={(e) => {
                            e.stopPropagation();
                            e.preventDefault();
                            showModal()
                        }}
                    />
                    <Modal title="Feedback" visible={isModalOpen} onOk={handleOk} onCancel={handleCancel} okText="Send" okButtonProps={{ loading: loading }}>
                        <InputFromAntd.TextArea
                            value={value}
                            onChange={(e) => setValue(e.target.value)}
                            placeholder="Write some feedback. I will read it (because I have to)."
                            autoSize={{ minRows: 3 }}
                        />
                    </Modal>
                </>
            );
        };

        const RateRender = (o) => {
            const [loading, setLoading] = useState(false);
            return <>
                <Col>
                    <Row style={{ marginBottom: 4 }}>
                        <Button
                            type="ghost"
                            icon={<HeartOutlined />}
                            loading={loading}
                            onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                setLoading(true)
                                callRate(o.o, 'good', o.type, () => setLoading(false), () => setLoading(false))
                            }}
                        />
                    </Row>
                    <Row style={{ marginBottom: 4 }}>
                        <Button
                            type="ghost"
                            icon={<MehOutlined />}
                            loading={loading}
                            onClick={(e) => {
                                e.stopPropagation();
                                e.preventDefault();
                                setLoading(true)
                                callRate(o.o, 'meeh', o.type, () => setLoading(false), () => setLoading(false))
                            }}
                        />
                    </Row>
                    <Row>
                        <ModalMessage o={o.o} type={o.type}></ModalMessage>
                    </Row>
                </Col>
            </>
        }

        const getSearch = () => {

            let input = new Input()
                .onChange((v) => this.setState({text: v}))
                .onPressEnter(() => {
                    let get = new Get()
                        .target(() => {
                            console.log('Enter')
                            return ({
                                target: `/request`,
                                params: {
                                    text: this.state.text,
                                    limit: this.state.limit ? this.state.limit : 0,
                                    max_id: this.state.maxId ? this.state.maxId : 0,
                                    predict: !!checkboxPredict._data,
                                    ignore: !!checkboxIgnore._data
                                }
                            })
                        })
                        .onComplete((v: any) => {
                            this.setState({workload: v.data.workload})
                            let data = v.data.data
                            if (data.length > 0) {
                                let dataAltered: any[] = []
                                data.forEach(r => {
                                    let tmp = r
                                    tmp.true_placement = r.workload.data.true_placement
                                    tmp.true_responsible = r.workload.data.true_responsible
                                    if (!checkboxIgnore._data || (tmp.true_placement == 'unknown' && tmp.true_responsible == 'unknown')) {
                                        dataAltered.push(tmp)
                                    }
                                })
                                condition.checkCondition({
                                    data: dataAltered,
                                    loading: false,
                                })
                            }
                            input.tsxSetDisabled(false)
                        });
                    get.get()
                    input.tsxSetDisabled(true)
                })

            return input
        };

        const getList = (args) => {
            return new List()
                .unique((v) => v.id)
                .default({dataSource: args})
                .footer(false)
                .expandableByClick()
                .headerCreate(false)
                .headerPrepend(new ListHeader().key('id').title('Request').width('136px').searchable())
                .headerPrepend(new ListHeader().key('subject').title('').searchable())
                .headerPrepend(new ListHeader().key('placement_rate').title('').width('40px').render(
                    (v, o) => <RateRender o={o} type='placement_rate'/>
                ))
                .headerPrepend(new ListHeader().key('placement').title('Placement').render(
                    (v, o) => {

                        let objects = [0, 1, 2].map(v => ({
                            name: o.workload.data.predict_placement[v].name,
                            value: o.workload.data.predict_placement[v].prediction_log,
                            color: '',
                            workload: 0,
                        }));

                        objects.forEach(v => {
                            v.workload = this.state.workload.placement[v.name]
                        })

                        let choice = objects[0];
                        objects.forEach(v => {
                            if (v.value >= limit_high && v.workload < choice.workload) {
                                choice = v;
                            }
                        })

                        objects.forEach(v => {
                            // v.color = v.name == choice.name ? 'primary' : (v.value > limit_high ? '#52c41a' : (v.value < limit_low ? '#f5222d' : '#faad14'))
                            v.color = v.value > limit_high ? '#52c41a' : (v.value < limit_low ? '#f5222d' : '#faad14')
                        })


                        return <>
                            <Col>
                                {
                                    objects.map(v => {
                                        return <Row>
                                            <Col>
                                                <Badge.Ribbon text={v.value.toFixed(1)} color={v.color}>
                                                    <Descriptions layout="horizontal" size="small" bordered
                                                              style={v.name == choice.name
                                                                  ? { width: 300, padding: 0, marginTop: 4, marginBottom: 4 }
                                                                  : { width: 300, padding: 0, marginTop: 4, marginBottom: 4 }
                                                              }>
                                                        <Descriptions.Item span={12} labelStyle={{width: 80}} label={<>
                                                            <Badge status={v.name == choice.name ? 'success' : 'default'} text={v.workload ?? 0} />
                                                        </>}>
                                                            {v.name}
                                                        </Descriptions.Item>
                                                    </Descriptions>
                                                </Badge.Ribbon>
                                            </Col>
                                        </Row>
                                    })
                                }
                            </Col>
                        </>
                    }
                ))
                .headerPrepend(new ListHeader().key('responsible_rate').title('').width('40px').render(
                    (v, o) => <RateRender o={o} type='responsible_rate'/>
                ))
                .headerPrepend(new ListHeader().key('responsible').title('Responsible').render(
                    (v, o) => {
                        return <>
                            <Col>
                                {
                                    [0, 1, 2].map(v => {
                                        let name = o.workload.data.predict_responsible[v].name;
                                        let value = o.workload.data.predict_responsible[v].prediction_log
                                        let color = value > limit_high ? '#52c41a' : (value < limit_low ? '#f5222d' : '#faad14')
                                        return <Row>
                                            <Col>
                                                <Badge.Ribbon text={value.toFixed(1)} color={color}>
                                                    <Descriptions layout="horizontal" size="small" bordered
                                                                  style={{width: 300, padding: 0, margin: 4}}>
                                                        <Descriptions.Item span={12} labelStyle={{width: 60}} label={<>
                                                            <Badge text={this.state.workload.responsible[name] ?? 0} />
                                                        </>}>
                                                            {name}
                                                        </Descriptions.Item>
                                                    </Descriptions>
                                                </Badge.Ribbon>
                                            </Col>
                                        </Row>
                                    })
                                }
                            </Col>
                        </>
                    }
                ))
                .expandable(() => true)
                .expandableExpandAll()
                .expandableSection((item: ListItem) => {

                    let section = new Section()

                    section.add(new Section().component(() => {

                        let placement = item._object.workload.data.true_placement;
                        let placement_log = item._object.predict.data.placement.find(e => e.name == placement)?.prediction_log ?? 0;
                        let responsible = item._object.workload.data.true_responsible;
                        let responsible_log = item._object.predict.data.responsible.find(e => e.name == responsible)?.prediction_log ?? 0;
                        let placement_color = placement_log > limit_high ? '#52c41a' : (placement_log < limit_low ? '#f5222d' : '#faad14')
                        let responsible_color = responsible_log > limit_high ? '#52c41a' : (responsible_log < limit_low ? '#f5222d' : '#faad14')

                        return <>
                            <Descriptions layout="horizontal" size="small" bordered
                                          style={{marginLeft: 4, marginRight: 4}}>
                                <Descriptions.Item span={12} label={<>
                                    <div style={{opacity: .7}}>Placement</div>
                                </>} labelStyle={{width: 136}}>
                                    {
                                        placement == 'unknown'
                                            ? ''
                                            : placement_log != 0
                                            ? <Badge color={placement_color} text={placement}/>
                                            : <Badge text={placement}/>
                                    }
                                </Descriptions.Item>
                                <Descriptions.Item span={12} label={<>
                                    <div style={{opacity: .7}}>Responsible</div>
                                </>} labelStyle={{width: 136}}>
                                    {responsible == 'unknown' ? '' : (responsible_log != 0 ?
                                        <Badge color={responsible_color} text={responsible}/> :
                                        <Badge text={responsible}/>)}
                                </Descriptions.Item>
                                <Descriptions.Item span={12} label={<>
                                    <div style={{opacity: .7}}>Request</div>
                                </>} labelStyle={{width: 136}}>
                                    {item._object.subject}
                                </Descriptions.Item>
                                <Descriptions.Item span={12} label={<></>}>
                                    <div dangerouslySetInnerHTML={{__html: item._object.description}}/>
                                </Descriptions.Item>
                            </Descriptions>
                        </>
                    }, false))

                    return section;
                });
        };

        const SelectMaxId = () => {

            const handleChange = (e) => {
                this.setState({ maxId: e.target.value })
            };

            return (
                <InputFromAntd
                    defaultValue="0"
                    style={{width: 120}}
                    onChange={handleChange}
                />
            )
        };

        const SelectLimit = () => {

            const handleChange = (value) => {
                this.setState({ limit: value })
            };

            return (
                <SelectFromAntd
                    defaultValue={this.state.limit}
                    style={{ width: 120 }}
                    onChange={handleChange}
                    options={[
                        { value: 10, label: '10' },
                        { value: 25, label: '25' },
                        { value: 50, label: '50' },
                    ]}
                />
            )
        };

        const main: Main = this.props.main;
        const search = getSearch();
        const section = new Section();

        let checkboxPredict = new Switch().data(true).toggleTrue('');
        let checkboxLatest = new Switch().data(true).toggleTrue('');
        let checkboxIgnore = new Switch().data(false).toggleTrue('');

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
        section.add(new Space().top(16));
        section.add(new Section().component(SelectLimit, false))
        section.add(new Section().component(() => <><div style={{ paddingTop: 4, paddingLeft: 0 }}>Limit the amount of results</div></>, false));
        // section.add(new Space().top(16));
        // section.add(new Section().component(SelectMaxId, false))
        // section.add(new Section().component(() => <><div style={{ paddingTop: 4, paddingLeft: 0 }}>The max-id being considered by the search</div></>, false));
        section.add(new Space().top(16));
        section.add(checkboxPredict)
        section.add(new Section().component(() => <><div style={{ paddingTop: 4, paddingLeft: 0 }}>Insure all results has predictions</div></>, false));
        section.add(new Space().top(16));
        section.add(checkboxIgnore)
        section.add(new Section().component(() => <><div style={{ paddingTop: 4, paddingLeft: 0 }}>Ignore all with placement or responsible</div></>, false));
        section.add(new Space().top(16));
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