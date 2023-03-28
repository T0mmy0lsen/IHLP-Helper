import React from "react";

import {
    ListHeader,
    Section,
    Space,
    Title,
    List,
    Main,
    ListItem, Conditions, ConditionsItem, Autocomplete, Get, Item, Slider,
} from 'react-antd-admin-panel/dist';

import {SectionComponent} from 'react-antd-admin-panel/dist';
import {message} from "antd";

interface Data {
    true_placement: string;
    true_responsible: string;
    true_timeconsumption: number;
    predict_placement: any;
    predict_responsible: number;
    predict_timeconsumption: number;
}

interface Request {
    priority: number;
    request_id: number;
    true_placement: string;
    true_responsible: string;
    true_timeconsumption: number;
    true_deadline: string;
    request: any;
    data: Data;
}

export default class Schedule extends React.Component<any, any> {

    constructor(props) {
        super(props);
        this.state = {
            section: false,
            slider: 1,
            data: [],
        }
    }

    build() {

        const main: Main = this.props.main;
        const section = new Section();
        const condition = new Conditions()

        function parseDeadline(deadline: string): Date {
            if (deadline == '1970-01-01T01:00:00') {
                let date = new Date();
                date.setDate(date.getDate() + 7);
                return date;
            }
            const [year, month, day, time] = deadline.split(/[-T:]/);
            return new Date(parseInt(year), parseInt(month) - 1, parseInt(day), parseInt(time));
        }

        function calculateRemainingTime(request: Request): number {
            return request.data.predict_timeconsumption - request.true_timeconsumption;
        }

        function leastTimeRemainingScheduling(requests: Request[], slider: number): Request[] {

            const hours = 24 * slider;  // Within 48 hours, the remaining time takes precedent.
            const deadlineScaler = (1/(60 * hours));
            const remainingTimeScaler = 1;

            requests.forEach((r, i) => {
                requests[i]['subject'] = r['request'].subject;
                requests[i]['remainingTime'] = Math.max(0.01, calculateRemainingTime(r) * remainingTimeScaler);
                requests[i]['timeToDeadline'] = ((parseDeadline(r.true_deadline).getTime() - new Date().getTime()) / (60 * 1000)) * deadlineScaler;  // Minutes
            })

            requests.forEach((r, i) => {
                requests[i]['priority'] = requests[i]['remainingTime'] + requests[i]['timeToDeadline']
                requests[i]['deadline'] = requests[i]['true_deadline'] == '1970-01-01T01:00:00' ? '' : requests[i]['true_deadline']
            })

            return requests.sort((a, b) => {
                return a.priority - b.priority;
            });
        }

        const getList = (requests: Request[]) => {
            return new List()
                .unique((v) => v.id)
                .default({ dataSource: requests })
                .footer(false)
                .expandableByClick()
                .headerCreate(false)
                .headerPrepend(new ListHeader().key('request_id').title('Request').width('136px').searchable())
                .headerPrepend(new ListHeader().key('subject').title('').searchable())
                .headerPrepend(new ListHeader().key('remainingTime').title('remainingTime').searchable())
                .headerPrepend(new ListHeader().key('timeToDeadline').title('timeToDeadline').searchable())
                .headerPrepend(new ListHeader().key('deadline').title('deadline').searchable())
                .expandable(() => false)
                .expandableExpandAll()
                .expandableSection((item: ListItem) => {

                    let section = new Section()

                    section.add(new Section().component(() => {

                        return <></>
                    }, false))

                    return section;
                });
        };

        const getCondition = (condition) => {
            return condition
                .default(() => ({ value: undefined }))
                .add(new ConditionsItem()
                    .condition((v: any) => {
                        return !!v.data
                    })
                    .content((next, callback, main, v) => {
                        next(new Section()
                            .add(new Space().top(24))
                            .add(getList(leastTimeRemainingScheduling(v.data, v.slider)))
                        )
                    })
                )
                .add(new ConditionsItem()
                    .condition((v: any) => !v.data)
                    .content((next) => next(new Section()))
                );
        }

        const getAutocomplete = () => {
            return new Autocomplete()
                .key('autocomplete')
                .label('Search')
                .get(() => new Get()
                    .target(() => '/placement')
                    .alter((v, args) => {
                        return v.data.placement.filter(r => r.name.includes(args)).map(r => new Item(r.name).title(r.name).text(r.name))
                    })
                    .onError(() => message.error('Error Get().target(() => \'placement\')'))
                )
                .onChange((v) => {
                    if (v.value) {
                        new Get()
                            .target(() => ({ target: '/schedule', params: { placement: v.object.key }}))
                            .onComplete((v) => {
                                if (v.status == 200) {
                                    this.setState({ data: v.data.data })
                                    condition.checkCondition({ data: v.data.data, slider: this.state.slider })
                                }
                            })
                            .onError(() => message.error('Error Get().target(() => \'schedule\')'))
                            .get()
                    }
                })
        }

        section.style({ padding: '24px 36px' });
        section.add(new Title().label('Create a Schedule').level(1));
        section.add(new Space().top(16));
        section.add(new Slider().default(this.state.slider).min(1).max(300).onChange((v) => {
            if (this.state.data.length > 0) {
                this.setState({ slider: v })
                condition.checkCondition({ data: this.state.data, slider: v })
            }
        }))
        section.add(new Space().top(32));
        section.add(getAutocomplete())
        section.add(getCondition(condition));

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